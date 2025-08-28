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
from search_space import get_config_space
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
from src.utils import ModelTooBig
import pandas as pd


pd.options.display.max_columns = None
app = Typer()

BASE_OUTPATH = Path(__file__).parent.parent / "chronos_models"
BAD_NODES_TRACKER = Path(__file__).parent.parent / "cache/broken_nodes.txt" # HPC isn't perfect some nodes are flawed and everything goes OOM on them -> Track & avoid
DB_PATH = Path(__file__).parent.parent / "AION.db"
OBJECTIVES = ["RMSE", "MASE", "WQL"]


@app.command()
@use_yaml_config(param_name="config")
def main(
    training_folder: str = Option("../data/train", help="Folder with all the Training Corpora (in .arrow format) that can be used"),
    seed: int = Option(0, help="Random seed for reproducibility."),
    model_ids: str = Option("['google/t5-efficient-tiny']", help="Which base model to use."),
    limit_model_size: int = Option(1, help= "Whether to limit model to about the size proposed by model_id or to let it grow indefinetly"),
    trial_walltime_limit: int = Option(300, help="How long until we stop a trial. (-1 ~ Unlimited)"),
    number_trials: int = Option(5, help="How many trials to run."),
    min_budget: int = Option(960_000, help="Minimum number of Training Samples"),
    max_budget: int = Option(512_000_000, help="Maximum number of Training Samples"),
    eta: int = Option(3, help="Hyperband eta."),
    memory: str = Option("160G", help="Memory to allocate for each worker."),
    worker_walltime: str = Option("01:00:00", help="How long shall each worker live."),
    account: str = Option("chronos_project", help="To which project shall the jobs be assigned. none for no project."),
    job_extra_directives: str = Option("['--gres=gpu:1']", help="Extra directives to be passed to the job scheduler."),
    worker_count: int = Option(1, help="How many workers to use."),
    max_batch_size: int = Option(32, help="How large is the max batch size per device. Note: Larger BS are simulated via Gradient Accumulation"),
):
    """Performance SMAC search for best Chronos Config"""

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
        death_timeout=600
    )

    print(cluster.job_script())
    cluster.scale(jobs=worker_count)  # Ask for 1 job
    client = Client(address=cluster)



    # Start with a few random configs + the default
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5) #, additional_configs=[configs_space.get_default_configuration()])

    # Create our intensifier
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget", seed=seed, eta=eta)

    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario=scenario,
        target_function=train, 
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
        dask_client=client,
        multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(
            scenario,
            objective_weights=[1, 0.5, 0.5],  # Equal Weights but MASE & WQL are largely redundant
        ),
    )

    # Load Checkpoints from previous Searches
    if DB_PATH.is_file():
        conn = sqlite3.connect(DB_PATH)

        # If Table Exists load prev trials from it
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Results';")
        table_exists = cur.fetchone() is not None
        if table_exists:
            print("Trying to add Previous Runs to History to learn from them...")
            df = pd.read_sql("SELECT * FROM Results", conn)
        else:
            df = pd.DataFrame()
        conn.close()

        # Add previous trials to our history if they fit the search space
        for _, row in df.iterrows():
            config: dict = json.loads(row["config"])
            config = {key: config[key] for key in configs_space.keys() if key in config}
            try: 
                smac._runhistory.add(
                    config=Configuration(
                        configuration_space=configs_space,
                        values=config
                        ),
                    cost=[row[metric] for metric in OBJECTIVES],
                    time=row["duration"],
                    cpu_time=row["duration"]*row["cpu_count"],
                    budget=row["budget"],
                    seed=row["seed"],
                )
            except IllegalValueError as e:
                print(f"Wasn't able to add Config:\n{config}\nbecause of: {str(e)}")

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
    budget: int = 1
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
        # Hash Config (incl. budget) to get exactly one model per config
        config_dict = dict(config)
        config_dict["seed"] = seed
        config_dict["batch_size"] = 2 ** config_dict["batch_size_expo"] 
        config_dict["per_device_train_batch_size"] = min(config_dict["max_per_device_train_batch_size"], config_dict["batch_size"])
        config_dict["num_devices"] = torch.cuda.device_count()
        real_bs = config_dict["per_device_train_batch_size"] * config_dict["num_devices"]
        config_dict["gradient_accumulation_steps"] = (config_dict["batch_size"] + (real_bs - 1)) // (real_bs) 
        training_steps = (int(budget) + config_dict["batch_size"] -1) // config_dict["batch_size"]  # Convert #Training Samples to number training steps
        config_dict["training_steps"] = training_steps
        config_hash = hash(frozenset([(key, str(val)) for key, val in config_dict.items()]))
        output_path =  Path(BASE_OUTPATH / f"chronos_{config_hash}")

        # if output_path.exists():
        #     raise NotImplementedError(f"Model already trained for config {config_hash}.")
        #     # TODO Load Config Results from DB and return (If already exists it should be in DB already so maybe todo?)
        #     return
        
        config: dict = dict(config)

        # Import Scripts to set their loggers (TODO make less dirty)
        from src.utils import make_dict_storable
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
            average_errors = {metric: float('inf') for metric in OBJECTIVES}



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
            **average_errors}, 
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
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    app()
