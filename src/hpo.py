# Include Parent Directory to load packages from
import sys  
import os  
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"src").resolve()))  
sys.path.append(str((Path(__file__).parent.parent / "chronos_pkg/src").resolve()))
from ConfigSpace import Configuration
from ConfigSpace.exceptions import IllegalValueError
from pathlib import Path
from typer import Typer, Option
from typer_config import use_yaml_config
import logging

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
from src.search_space import get_config_space
from smac import HyperparameterOptimizationFacade as HPOFacade
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import uuid
from ast import literal_eval
import torch
import pandas as pd
import socket
import sqlite3
import json
import time
from datetime import datetime
from src.db import insertTable
from src.utils import ModelTooBig, make_dict_storable
import pandas as pd
from math import ceil
import math
import types
from typing import List
from functools import partial
from smac.runhistory.dataclasses import TrialValue, TrialInfo
from smac.runhistory.enumerations import StatusType
from smac.main.config_selector import ConfigSelector
from collections import defaultdict


pd.options.display.max_columns = None
app = Typer(pretty_exceptions_enable=False)

BASE_OUTPATH = Path(__file__).parent.parent / "chronos_models"
BAD_NODES_TRACKER = Path(__file__).parent.parent / "cache/broken_nodes.txt" # HPC isn't perfect some nodes are flawed and everything goes OOM on them -> Track & avoid
OBJECTIVES = ["RMSE", "MAE", "WQL"]
PREFERENCE_FUNCTION = {
    "RMSE": 1, 
    "MAE": 1, 
    "WQL": 1,
}

def map_int_to_nearest_float(x: int, values: List[float]) -> float:
    return min(values, key=lambda v: abs(v - x))

def get_checkpoints_from_db(DB_PATH: Path):
    if DB_PATH.is_file():
        conn = sqlite3.connect(DB_PATH)

        # If Table Exists load prev trials from it
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Results';")
        table_exists = cur.fetchone() is not None
        if table_exists:
            print("Trying to add Previous Runs to History to learn from them...")
            df = pd.read_sql("SELECT * FROM Results ORDER BY budget ASC", conn)
        else:
            df = pd.DataFrame()
        conn.close()
        return df
    return pd.DataFrame()


def update_tracker(smac: MFFacade, checkpointed_trials: list):
    """
    Tells Smac to first evaluate already conducted trials. 
    Else it uses checkpointed trials only for training but allocates budget starting at bracket 1 

    Returns:
        _type_: _description_
    """
    # TODO Fix Bug to also support meta learning/warmstarting
    next(smac._optimizer._trial_generator)
    # Calculate Offset for case that we completed an entire HB loop already
    trial_count_per_fidelity = defaultdict(int)
    max_budget = smac._optimizer.intensifier._max_budget
    for bracket, max_stage in smac._optimizer.intensifier._max_iterations.items():
        # budget = smac._optimizer.intensifier._get_instance_seed_budget_keys_by_stage(bracket=bracket, stage=max_stage-1, seed=seed)
        for stage in range(max_stage):
            trial_count = smac._optimizer.intensifier._n_configs_in_stage[bracket][stage]
            budget = smac._optimizer.intensifier._get_instance_seed_budget_keys_by_stage(bracket=bracket, stage=stage)[0].budget
            trial_count_per_fidelity[budget] += trial_count
    q = len([checkpoint for checkpoint in checkpointed_trials if checkpoint.budget == max_budget]) // trial_count_per_fidelity[max_budget]

    checkpointed_trials_dict = defaultdict(list)
    for trial in checkpointed_trials:
        checkpointed_trials_dict[trial.budget].append(trial.config)

    # Skip artifacts of first q runs
    for budget in checkpointed_trials_dict.keys():
        startpoint = q * trial_count_per_fidelity[budget]
        assert startpoint <= len(checkpointed_trials_dict[budget])
        checkpointed_trials_dict[budget] = checkpointed_trials_dict[budget][startpoint:]

    # Checkpoints are cleaned --> insert into tracker
    trials_assigned = defaultdict(list)
    for budget, trials in checkpointed_trials_dict.items():
        # fill up each stage bracket by bracket
        for i, trial in enumerate(trials):
            bracket = 0
            stage = smac._optimizer.intensifier._budgets_in_stage[bracket].index(budget)
            trial_count = smac._optimizer.intensifier._n_configs_in_stage[bracket][stage]
            while i >= trial_count:
                i -= trial_count 
                bracket += 1
                if budget in smac._optimizer.intensifier._budgets_in_stage[bracket]:
                    stage = smac._optimizer.intensifier._budgets_in_stage[bracket].index(budget)
                    trial_count = smac._optimizer.intensifier._n_configs_in_stage[bracket][stage]
                else: 
                    break
            trials_assigned[(bracket,stage)].append(trial)
    for key, trials in trials_assigned.items():
        if len(smac.intensifier._tracker[key]) > 0:
            smac.intensifier._tracker[key][0][1][:len(trials)] = trials
        else:
            smac.intensifier._tracker[key].append((0, trials))

            




@app.command()
@use_yaml_config(param_name="config")
def main(
    training_folder: str = Option("./data/train", help="Folder with all the Training Corpora (in .arrow format) that can be used"),
    db_name: str = Option("AION.db", help="Name of the sqlite database to write to"),
    seed: int = Option(0, help="Random seed for reproducibility. -1 for choosing one at random - recommended when starting multiple mains in parallel that communicate via checkpoints (see migration model for evolutionary algorithms)."),
    model_ids: str = Option("['google/t5-efficient-tiny']", help="Which base model to use."),
    limit_model_size: int = Option(1, help= "Whether to limit model to about the size proposed by model_id or to let it grow indefinetly. Set to two for preemptively stopping them and not even counting them"),
    checkpoint_broken_trials: int = Option(1, help= "When restarting from Checkpoint shall aborted/crashed runs be reloaded into history?"),
    trial_walltime_limit: int = Option(300, help="How long until we stop a trial. (-1 ~ Unlimited)"),
    number_trials: int = Option(5, help="How many trials to run."),
    min_budget: int = Option(960_000, help="Minimum number of Training Samples"),
    max_budget: int = Option(512_000_000, help="Maximum number of Training Samples"),
    eta: float = Option(3, help="Hyperband eta."),
    memory: str = Option("160G", help="Memory to allocate for each worker."),
    worker_walltime: str = Option("01:00:00", help="How long shall each worker live."),
    account: str = Option("chronos_project", help="To which project shall the jobs be assigned. none for no project."),
    job_extra_directives: str = Option("['--gres=gpu:1']", help="Extra directives to be passed to the job scheduler."),
    worker_count: int = Option(1, help="How many workers to use."),
    max_batch_size: int = Option(32, help="How large is the max batch size per device. Note: Larger BS are simulated via Gradient Accumulation"),
    checkpointing: str = Option("official", help="Which Checkpointing Mode to use 'official' for save behaviour, 'db' to load previous trials from database, 'none' to start from scratch"),
):
    """Performs SMAC search for best Chronos Config"""
    DB_PATH = Path(__file__).parent.parent / db_name
    checkpoints = get_checkpoints_from_db(DB_PATH=DB_PATH)

    if seed == -1:
        # Generate Random Seed
        seed = int.from_bytes(os.urandom(8), "big")
        seed ^= int(time.time_ns())


    # Get Search Space
    configs_space = get_config_space(training_folder=training_folder, model_ids=model_ids, max_batch_size=max_batch_size, limit_model_size=limit_model_size)

    # Define environment variables
    scenario = Scenario(
        configspace=configs_space,
        name=f"{literal_eval(model_ids)[0].replace('/', '_')}_{uuid.uuid4()}",
        output_directory=root_dir / "hpc/logs/smac3_output",
        trial_walltime_limit=trial_walltime_limit if trial_walltime_limit > 0 else None,  
        n_trials=number_trials,  
        min_budget=min_budget,  
        max_budget=max_budget,  
        seed=seed,
        use_default_config=True,
        objectives=OBJECTIVES,  
        n_workers=worker_count
    )

    # Some Nodes on HPC are sometimes broken --> we try to track and avoid them
    if BAD_NODES_TRACKER.exists():
        with BAD_NODES_TRACKER.open() as f:
            bad_nodes = [line.strip() for line in f if line.strip()]
            bad_nodes = ",".join(sorted(set(bad_nodes)))
    else:
        bad_nodes = ""
    print(bad_nodes)

    # Submit Worker Jobs
    cluster = SLURMCluster(
        job_cpu=4, 
        cores=1,
        memory=memory,
        walltime=worker_walltime,
        job_extra_directives=literal_eval(job_extra_directives)+[(f"--exclude={bad_nodes}" if bad_nodes else "")],
        account=account if account.lower() != "none" else None,
        job_name="chronos_hpo",
        local_directory=str((root_dir / "hpc/logs/dask").resolve()),
        processes=1,
        log_directory=str((root_dir / "hpc/logs/smac").resolve()),
        nanny = False,
        job_script_prologue=[
            f"cd {root_dir.resolve()}",
            "pwd",
            "nvidia-smi",
            "source ./hpc/modules.sh",
            "which python",
            "python --version",
        ],
        death_timeout=None # TODO Check if works
    )

    print(cluster.job_script())
    cluster.scale(jobs=worker_count)  # Ask for 1 job
    client = Client(address=cluster)

    # Start with a few random configs + the default
    seed_set_size = worker_count*2
    if len(checkpoints) < seed_set_size:
        initial_design = MFFacade.get_initial_design(scenario, n_configs=seed_set_size-len(checkpoints)) 
    else:
        initial_design = None

    # Create our intensifier
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget", seed=seed, eta=eta) # ! Must be highest budget others have bug (I think)

    config_selector = ConfigSelector(
        scenario=scenario,
        retrain_after=8,
        retries=16,
        min_trials=16, # Assumption longer training behaves very stably --> can rely on lower budgets for longer
    )

    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario=scenario,
        target_function=partial(train, DB_PATH=DB_PATH), 
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
        dask_client=client,
        config_selector=config_selector,
        multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(
            scenario,
            objective_weights=[PREFERENCE_FUNCTION[obj] for obj in OBJECTIVES],  # Equal Weights 
        ),
    )

    # Load Checkpoints from previous Searches
    if len(checkpoints) > 0:
        max_it = intensifier._get_max_iterations(eta, max_budget, min_budget)
        budgets, _ = intensifier._compute_configs_and_budgets_for_stages(
                eta, max_budget, max_it, max_it
            )
        print("Budgets:", budgets)

        # Add previous trials to our history if they fit the search space
        checkpointed_trials = []
        for _, row in checkpoints.iterrows():
            config: dict = json.loads(row["config"])
            config = {key: config[key] for key in configs_space.keys() if key in config}

            try: 
                if not checkpoint_broken_trials and any(math.isinf(row[metric]) for metric in OBJECTIVES):
                    # TODO Check if this leads to same number broken 1,528/2,330 (Hypothesis: RF learns to avoid inf -> more broken when disabled)
                    continue
                info = TrialInfo(
                    config = Configuration(
                                configuration_space=configs_space,
                                values=config
                            ),
                    instance = None,
                    seed = row["seed"],
                    budget = map_int_to_nearest_float(row["budget"], budgets), # we track budget as ints, smac as floats
                )
                value = TrialValue(
                            cost = [row[metric] for metric in OBJECTIVES],
                            time = row["duration"],
                            cpu_time = row["duration"]*row["cpu_count"],
                            status = StatusType.SUCCESS,
                            starttime = 0.0,
                            endtime = 0.0,
                            additional_info = {}
                )
                smac.tell(
                    info=info,
                    value=value
                )
                
                checkpointed_trials.append(info)
            except IllegalValueError as e:
                print(f"Wasn't able to add Config:\n{config}\nbecause of: {str(e)}")

        # Update tracker to resume counting
        update_tracker(smac=smac, checkpointed_trials=checkpointed_trials)
        print("Budgets in History: ", {run_key.budget for run_key in smac._config_selector._runhistory_encoder._runhistory if run_key.budget is not None})
        x, y, confs = smac._config_selector._collect_data()
        print("Dimensions: ", x.shape, y.shape, confs.shape, flush=True)

    # Wait to get Scheduled
    client.wait_for_workers(1) 

    # Start The Search
    incumbents = smac.optimize()
    print(f"Incumbents: {incumbents}")


    for incumbent in incumbents:
        print(f"Best configuration: {incumbent}")
        # cost = smac.validate(incumbent)
        # print("---", cost)

    client.close()
    cluster.close()
    print(f"All Done!!! {len(smac.runhistory.get_configs("cost"))} Evaluated")
    print(f"You can find all the trials and their results in {DB_PATH}")




def train(
    config: Configuration, 
    seed: int = 0, 
    budget: int = 1,
    DB_PATH: Path = None,
    ):
    """
    The actual Trial 
    Trains and evaluates a Chronos Model with the given Config
    Writes the detailed results to the Database and returns the error metric as a dict

    Args:
        config (Configuration): An Hyperparameter Configuration from the given Search Space
        seed (int, optional): Random Seed that shall be used. Defaults to 0.
        budget (int, optional): For how many train samples shall the model be trained. Defaults to 1.

    Returns:
        _type_: _description_
    """
    assert DB_PATH is not None

    start_time = time.time()
    if torch.cuda.is_available():
        # Track Resource Consumption for Debugging CUDA
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        free = reserved - allocated

        print(f"Total: {total / 1e9:.2f} GB")
        print(f"Reserved: {reserved / 1e9:.2f} GB")
        print(f"Allocated: {allocated / 1e9:.2f} GB")
        print(f"Free (inside reserved): {free / 1e9:.2f} GB")

    try:
        budget = ceil(budget)
        print("Budget: ", budget)
        # Hash Config (incl. budget) to get exactly one model per config
        config_dict = dict(config)
        config_dict["seed"] = seed
        config_dict["batch_size"] = 2 ** config_dict["batch_size_expo"] 
        config_dict["num_devices"] = torch.cuda.device_count()
        config_dict["per_device_train_batch_size"] = min(config_dict["max_per_device_train_batch_size"], int(config_dict["batch_size"]//config_dict["num_devices"]))
        real_bs = config_dict["per_device_train_batch_size"] * config_dict["num_devices"]
        config_dict["gradient_accumulation_steps"] = (config_dict["batch_size"] + (real_bs - 1)) // (real_bs) 
        training_steps = (budget + config_dict["batch_size"] -1) // config_dict["batch_size"]  # Convert #Training Samples to number training steps
        config_dict["training_steps"] = training_steps
        config_hash = hash(frozenset([(key, str(val)) for key, val in config_dict.items()]))
        output_path =  Path(BASE_OUTPATH / f"chronos_{config_hash}")

        # if output_path.exists():
        #     raise NotImplementedError(f"Model already trained for config {config_hash}.")
        #     # TODO Load Config Results from DB and return (If already exists it should be in DB already so maybe todo?)
        #     return
        
        config: dict = dict(config)

        # Import Scripts to set their loggers (TODO make less dirty)
        import src.train as trainer
        import src.evaluate as evaluater
        # Set Missing Global Variables
        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__file__)
        logger.setLevel(logging.INFO)
        trainer.logger = logger
        evaluater.logger = logger

        # Compute Complex HPs (especially parameters that are powers of 2)
        d_kv = 2 ** config.pop("d_kv_expo", 6)
        d_ff = 2 ** config.pop("d_ff_expo", 12)
        context_length = 2 ** config.pop("context_length_expo", 9)
        prediction_length = 2 ** config.pop("prediction_length_expo", 6)
        d_model = 2 ** config.pop("d_model_expo", 9)
        min_past = 2 ** config.pop("min_past_expo", 6)
        bolt = True if config.pop("bolt", 0) else False
        model_type = "causal" if "gpt" in config["model_id"] else "seq2seq" # TODO expand
        limit_model_size = bool(config.pop("limit_model_size", 1))
        is_gated_act = bool(config.pop("is_gated_act", 0))
        if bolt:
            config["patch_stride"] = 2 ** config.pop("patch_stride_expo", 4)
            config["patch_size"] = 2 ** config.pop("patch_size_expo", 4)
            config["use_reg_token"] = True if config.pop("use_reg_token", 1) else False
        else:
            config["n_tokens"] = 2 ** config.pop("n_tokens_expo", 9)
            tokenizer_limit = config.pop("tokenizer_limit", 15)
            config["tokenizer_kwargs"] = f"{{'low_limit': -{tokenizer_limit:.3f}, 'high_limit': {tokenizer_limit:.3f}}}"
            print(f"Tokenizer Kwargs: {config["tokenizer_kwargs"]}")

        # Update/Remove Complex HPs that were calculated at beginning for experiment tracking
        config.pop("batch_size_expo")
        config.pop("max_per_device_train_batch_size")
        config["per_device_train_batch_size"] = config_dict["per_device_train_batch_size"]
        config["gradient_accumulation_steps"] = config_dict["gradient_accumulation_steps"]

        # Get Corpus Probabilities (s.t. we can optimize the training data)
        training_data_paths = config.pop("training_data_paths")
        training_data_paths_list = literal_eval(training_data_paths)
        probabilities = [config.pop(Path(p).stem) for p in training_data_paths_list]
        # Normalize
        total = sum(probabilities)
        probabilities = [prob/total for prob in probabilities]


        # Train Chronos and Save to outpath
        try:
            model_path = trainer.main(
                output_dir=output_path,
                training_data_paths=training_data_paths,
                probability=probabilities,
                max_steps=training_steps,
                d_kv=d_kv,
                d_ff=d_ff,
                context_length=context_length,
                prediction_length=prediction_length,
                d_model=d_model,
                bolt=bolt,
                min_past=min_past,
                model_type=model_type,
                limit_model_size=limit_model_size,
                is_gated_act=is_gated_act,
                **config, 
            )


            # Eval Chronos
            print(f"Start Evaluating {model_path}")
            val_config = Path("./data/eval_configs/eval_config.yml").resolve()
            results = evaluater.main(
                config_path=val_config,
                chronos_model_id = model_path,
                device="cuda",
                torch_dtype="bfloat16",
                batch_size=32,
                num_samples=20,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                bolt=bolt,
            )
            # Aggregate Results
            numeric_cols = results.select_dtypes(include='number').columns
            average_errors = results[numeric_cols].mean(numeric_only=True).to_dict()
            print(f"Results:\n {results}")
        except ModelTooBig as e:
            print(str(e), flush=True)
            # Rember Which models were too big - to train the surrogate model to not sample them over and over
            average_errors = {metric: float('inf') for metric in OBJECTIVES + ["MASE", "SMAPE", "Utility"]} # TODO don't hard code MAE and SMAPE



        duration = time.time() - start_time

        # Save Meta Data To DB
        config_simple = make_dict_storable(config_dict)
        insertTable("Results", {
            "time_stamp": datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f"),
            "config_hash":config_hash, 
            "config":config_simple, 
            "ModelPath":output_path, 
            "budget": budget, 
            "seed": seed, 
            "duration": duration, 
            "cpu_count": os.cpu_count(),
            "gpu_count": torch.cuda.device_count(),
            "Utility": sum([PREFERENCE_FUNCTION[obj] * average_errors[obj] for obj in OBJECTIVES]) / sum(PREFERENCE_FUNCTION.values()),
            **average_errors
            }, 
            db_path=DB_PATH,
            )

        # Remove current node from broken nodes list (might be obsolete)
        if BAD_NODES_TRACKER.exists():
            node_name = socket.gethostname() # Node worked at least ones -> Should not be broken
            lines = BAD_NODES_TRACKER.read_text().splitlines()
            lines = [line for line in lines if line.strip() != node_name]
            BAD_NODES_TRACKER.write_text("\n".join(lines) + "\n" if lines else "")

        # ! Return Costs
        return {key: average_errors[key] for key in OBJECTIVES} 

    except RuntimeError as e: 
        # Catch OOM Error seems to be a HW problem (they appear in swarms on the same device - Unlikely Config Specific)
        if "out of memory" in str(e) or "uncorrectable ECC error encountered" in str(e):
            # Track Broken Nodes
            node_name = socket.gethostname()
            print(f"OOM or ECC error caught on {node_name}: {str(e)}")
            with BAD_NODES_TRACKER.open("a") as f:
                f.write(f"{node_name}\n")
            torch.cuda.empty_cache()
        else:
            raise



if __name__ == "__main__":
    # Go to current file directory
    os.chdir(Path(__file__).resolve().parent.parent)

    app()
