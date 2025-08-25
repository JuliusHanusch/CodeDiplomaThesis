# Include Parent Directory to load packages from
import sys  
import os  
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"code").resolve()))  
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
import pandas as pd

pd.options.display.max_columns = None

BASE_OUTPATH = Path(__file__).parent.parent / "chronos_models"
BAD_NODES_TRACKER = Path(__file__).parent.parent / "cache/broken_nodes.txt" # HPC isn't perfect some nodes are flawed and everything goes OOM on them -> Track & avoid
DB_PATH = Path(__file__).parent.parent / "AION.db"
OBJECTIVES = ["RMSE", "MASE", "WQL"]

app = Typer()

@app.command()
@use_yaml_config(param_name="config")
def main(
    training_folder: str = Option("../data/train", help="Folder with all the Training Corpora (in .arrow format) that can be used"),
    seed: int = Option(0, help="Random seed for reproducibility."),
    model_ids: str = Option("['google/t5-efficient-tiny']", help="Which base model to use."),
    limit_model_size: int = Option(1, "Whether to limit model to about the size proposed by model_id or to let it grow indefinetly"),
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

    configs_space = get_config_space(training_folder=training_folder, model_ids=model_ids, max_batch_size=max_batch_size, limit_model_size=limit_model_size)


    # Define our environment variables
    scenario = Scenario(
        configspace=configs_space,
        name=f"{literal_eval(model_ids)[0].replace('/', '_')}_{uuid.uuid4()}",
        output_directory=root_dir / "hpc/logs/smac3_output",
        trial_walltime_limit=trial_walltime_limit if trial_walltime_limit > 0 else None,  
        n_trials=number_trials,  
        min_budget=min_budget,  
        max_budget=max_budget,  
        # n_workers=8, not by cluster jobs
        seed=seed,
        use_default_config=True,
        objectives=OBJECTIVES,  
    )

    if BAD_NODES_TRACKER.exists():
        with BAD_NODES_TRACKER.open() as f:
            bad_nodes = [line.strip() for line in f if line.strip()]
            # if len(set(bad_nodes)) > 10: # Some OOMs might be legit - test them again ever so often
            #     bad_nodes = random.sample(bad_nodes, len(bad_nodes)//2) # The more often OOM hapens the more likely one will be sampled
            bad_nodes = ",".join(sorted(set(bad_nodes)))
    else:
        bad_nodes = ""


    print(bad_nodes)
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
        death_timeout=300
    )

    print(cluster.job_script())

    cluster.scale(jobs=worker_count)  # Ask for 1 job

    client = Client(address=cluster)

    # We start with a few random configs + the default
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
            objective_weights=[1, 0.5, 0.5],  # Equal Weights but MASE & WQL are redundant
        ),
    )

    # Checkpointing from previous Searches
    if DB_PATH.is_file():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM Results", conn)
        for _, row in df.iterrows():
            config: dict = json.loads(row["config"])
            config = {key: config[key] for key in configs_space.keys() if key in config}
            try: # Search Spaces might differ over time - reuse only the currently relevant trials
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

    client.wait_for_workers(1) # Wait to get Scheduled

    # Let's optimize
    # print(f"History: {smac.runhistory.get_configs("cost")}")
    incumbents = smac.optimize()
    print(f"Incumbents: {incumbents}")
    # print(f"History: {smac.runhistory.get_configs("cost")}")


    for incumbent in incumbents:
        print(f"Best configuration: {incumbent}")
        # cost = smac.validate(incumbent)
        # print("---", cost)

    client.close()
    cluster.close()
    print(f"All Done!!! {len(smac.runhistory.get_configs("cost"))} Evaluated")




def train(
    config: Configuration, 
    seed: int = 0, 
    budget: int = 1
    ):
    start_time = time.time()
    if torch.cuda.is_available():
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
        # Hash Config to get exactly one model per config
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

        if output_path.exists():
            raise NotImplementedError(f"Model already trained for config {config_hash}.")
            # TODO Load Config Results from DB and return (If already exists it should be in DB already so maybe todo?)
            return
        
        config: dict = dict(config)

        # Import Train Function
        #from code.evaluate import main as eval_chronos
        from src.utils import make_dict_storable
        import src.train as trainer
        import src.evaluate as evaluater
        # Set Missing Global Variables
        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__file__)
        logger.setLevel(logging.INFO)
        trainer.logger = logger
        evaluater.logger = logger

        # Special HPs
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

        config.pop("batch_size_expo")
        config.pop("max_per_device_train_batch_size")
        config["per_device_train_batch_size"] = config_dict["per_device_train_batch_size"]
        config["gradient_accumulation_steps"] = config_dict["gradient_accumulation_steps"]

        # get ds probabilities in the according order
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


            # ! Eval Chronos
            val_configs = (
                Path("./data/eval_configs/eval_config.yml").resolve(), 
            )
            results = {}

            print(f"Start Evaluating {model_path} on {val_configs}")
            for val_config in val_configs: # TODO support multiple val configs
                results[val_config.stem]: pd.DataFrame = evaluater.main(
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
                numeric_cols = results[val_config.stem].select_dtypes(include='number').columns
                average_errors = results[val_config.stem][numeric_cols].mean(numeric_only=True).to_dict()
            print(f"Results:\n {results}")
        except Exception as e:
            if "ModelTooBig" in str(e) and DB_PATH.exists(): # Dont make Inf first entry that gets messy
                # Note Which models were too big to not sample them over and over
                average_errors = {metric: float('inf') for metric in OBJECTIVES}
                print(str(e))
            else: 
                raise

        duration = time.time() - start_time

        # ! Save Meta Data To DB
        config_simple = make_dict_storable(config_dict)
        # in_domain_mase, in_domain_wql, in_domain_mae, in_domain_nrmse, zero_shot_mase, zero_shot_wql, zero_shot_mae, zero_shot_nrmse = results_to_metrics(results)
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
            db_path=DB_PATH)
        if BAD_NODES_TRACKER.exists():
            node_name = socket.gethostname() # Node worked at least ones -> Should not be broken
            lines = BAD_NODES_TRACKER.read_text().splitlines()
            lines = [line for line in lines if line.strip() != node_name]
            BAD_NODES_TRACKER.write_text("\n".join(lines) + "\n" if lines else "")
        # ! Return Costs
        return {key: average_errors[key] for key in OBJECTIVES} #(average_errors["RMSE"], average_errors["MASE"], average_errors["WQL"])
    except RuntimeError as e: # Catch OOM Error seems to be a HW problem (they appear in swarms on the same device - Unlikely Config Specific)
        if "out of memory" in str(e) or "uncorrectable ECC error encountered" in str(e):
            # Get Broken Node
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


    # # TODO
    # main(
    #     config={}, # TODO Add Config
    #     training_data_paths = "['/data/horse/ws/jipo020b-aion/AION/data/testfiles/training_mix.arrow']",
    #     seed=0,
    #     model_ids='["google/t5-efficient-tiny"]',
    #     trial_walltime_limit=-1,
    #     number_trials=6,
    #     min_budget=512,
    #     max_budget=2024,
    #     eta=3,
    #     memory="160G",
    #     worker_walltime="02:00:00",
    #     account="p_automl",
    #     job_extra_directives="['--gres=gpu:1']",
    #     worker_count=1,
    #     max_batch_size=1
    # )
    #app()