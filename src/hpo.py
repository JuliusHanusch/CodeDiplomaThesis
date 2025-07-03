# Include Parent Directory to load packages from
import sys  
import os  
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir.resolve()))  
sys.path.append(str((root_dir/"code").resolve()))  
sys.path.append(str((Path(__file__).parent.parent / "chronos_pkg/src").resolve()))
from ConfigSpace import Configuration, ConfigurationSpace
from functools import partial
from pathlib import Path
from typer import Typer, Option
from typer_config import use_yaml_config
import logging

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
from search_space import get_config_space, training_data_paths
from smac import HyperparameterOptimizationFacade as HPOFacade
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from time import sleep
import uuid
from ast import literal_eval
import torch
import pandas as pd
import socket
import random

BASE_OUTPATH = Path("./chronos_models")
BAD_NODES_TRACKER = Path("./cache/broken_nodes.txt") # HPC isn't perfect some nodes are flawed and everything goes OOM on them -> Track & avoid

app = Typer()

@app.command()
@use_yaml_config(param_name="config")
def main(
    seed: int = Option(0, help="Random seed for reproducibility."),
    model_id: str = Option("google/t5-efficient-tiny", help="Which base model to use."),
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

    configs_space = get_config_space(model_id=model_id, max_batch_size=max_batch_size)


    # Define our environment variables
    scenario = Scenario(
        configspace=configs_space,
        name=f"{model_id.replace('/', '_')}_{uuid.uuid4()}",
        output_directory=root_dir / "hpc/logs/smac3_output",
        trial_walltime_limit=trial_walltime_limit if trial_walltime_limit > 0 else None,  
        n_trials=number_trials,  
        min_budget=min_budget,  
        max_budget=max_budget,  
        # n_workers=8, not by cluster jobs
        seed=seed,
        use_default_config=True,
        objectives=["RMSE", "MASE", "WQL"],  
    )

    if BAD_NODES_TRACKER.exists():
        with BAD_NODES_TRACKER.open() as f:
            bad_nodes = [line.strip() for line in f if line.strip()]
            if len(bad_nodes) > 10: # Some OOMs might be legit - test them again ever so often
                bad_nodes = random.sample(bad_nodes, len(bad_nodes)//2) # The more often OOM hapens the more likely one will be sampled
            bad_nodes = ",".join(sorted(set(bad_nodes)))
    else:
        bad_nodes = ""

    # TODO
    cluster = SLURMCluster(
        job_cpu=4, #TODO Can we increase to 4
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

    sleep(35)

    # Let's optimize
    incumbents = smac.optimize()
    print(f"Incumbents: {incumbents}")
    print(f"History: {smac.runhistory}")


    for incumbent in incumbents:
        print(f"Best configuration: {incumbent}")
        cost = smac.validate(incumbent)
        print("---", cost)

    client.close()
    cluster.close()
    print("All Done!!!")




def train(
    config: Configuration, 
    seed: int = 0, 
    budget: int = 1
    ):
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
        metrics_to_optimize = ["RMSE", "MASE", "WQL"]

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
            # TODO Load Config Results from DB and return
            return
        
        config: dict = dict(config)

        # Import Train Function
        #from code.evaluate import main as eval_chronos
        from src.utils import make_dict_storable, results_to_metrics
        import src.train as trainer
        import src.evaluate as evaluater
        # Set Missing Global Variables
        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__file__)
        logger.setLevel(logging.INFO)
        trainer.logger = logger
        evaluater.logger = logger

        # Special HPs
        tokenizer_limit = config.pop("tokenizer_limit", 15)
        tokenizer_kwargs = f"{{'low_limit': -{tokenizer_limit:.3f}, 'high_limit': {tokenizer_limit:.3f}}}"
        d_kv = 2 ** config.pop("d_kv", 6)
        d_ff = 2 ** config.pop("d_ff", 12)
        context_length = 2 ** config.pop("context_length", 9)
        prediction_length = 2 ** config.pop("prediction_length", 6)
        d_model = 2 ** config.pop("d_model", 9)
        n_tokens = 2 ** config.pop("n_tokens", 9)
        print(f"Tokenizer Kwargs: {tokenizer_kwargs}")
        config.pop("batch_size_expo")
        config.pop("max_per_device_train_batch_size")
        config["per_device_train_batch_size"] = config_dict["per_device_train_batch_size"]
        config["gradient_accumulation_steps"] = config_dict["gradient_accumulation_steps"]

        # Train Chronos and Save to outpath
        model_path = trainer.main(
            output_dir=output_path,
            training_data_paths=training_data_paths,
            max_steps=training_steps,
            tokenizer_kwargs=tokenizer_kwargs,
            d_kv=d_kv,
            d_ff=d_ff,
            context_length=context_length,
            prediction_length=prediction_length,
            d_model=d_model,
            n_tokens=n_tokens,
            **config, 
        )


        # ! Eval Chronos
        val_configs = (
            Path("./data/eval_configs/eval_config.yml").resolve(), # TODO Path to Zero-Shot Val Config
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
            )
            # Aggregate Results
            numeric_cols = results[val_config.stem].select_dtypes(include='number').columns
            average_errors = results[val_config.stem][numeric_cols].mean(numeric_only=True).to_dict()

        # TODO Check that all metrics are near 0-1 range (e.g. NRMSE)
        import pandas as pd
        pd.options.display.max_columns = None
        print(f"Results:\n {results}")

        # ! Save Meta Data To DB

        from src.db import insertTable

        config_simple = make_dict_storable(config_dict)
        # in_domain_mase, in_domain_wql, in_domain_mae, in_domain_nrmse, zero_shot_mase, zero_shot_wql, zero_shot_mae, zero_shot_nrmse = results_to_metrics(results)
        insertTable("Results", {"config_hash":config_hash, "config":config_simple, "ModelPath":output_path, **average_errors})
                                # "in_domain_mase":in_domain_mase, "in_domain_wql":in_domain_wql, "in_domain_mae":in_domain_mae, "in_domain_nrmse":in_domain_nrmse,
                                # "zero_shot_mase":zero_shot_mase,"zero_shot_wql":zero_shot_wql,"zero_shot_mae":zero_shot_mae,"zero_shot_nrmse":zero_shot_nrmse })
        if BAD_NODES_TRACKER.exists():
            node_name = socket.gethostname() # Node worked at least ones -> Should not be broken
            lines = BAD_NODES_TRACKER.read_text().splitlines()
            lines = [line for line in lines if line.strip() != node_name]
            BAD_NODES_TRACKER.write_text("\n".join(lines) + "\n" if lines else "")
        # ! Return Costs
        return {key: average_errors[key] for key in metrics_to_optimize} #(average_errors["RMSE"], average_errors["MASE"], average_errors["WQL"])
    except RuntimeError as e: # Catch OOM Error seems to be a HW problem (they appear in swarms on the same device - Unlikely Config Specific)
        if "out of memory" in str(e):
            # Get Broken Node
            node_name = socket.gethostname()
            print(f"OOM error caught on {node_name}")
            with BAD_NODES_TRACKER.open("a") as f:
                f.write(f"{node_name}\n")
            torch.cuda.empty_cache()
        else:
            raise



if __name__ == "__main__":
    # Go to current file directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


    # TODO
    main(
        config={}, # TODO Add Config
        seed=0,
        model_id="google/t5-efficient-tiny",
        trial_walltime_limit=-1,
        number_trials=6,
        min_budget=512,
        max_budget=2024,
        eta=3,
        memory="160G",
        worker_walltime="02:00:00",
        account="p_automl",
        job_extra_directives="['--gres=gpu:1']",
        worker_count=4,
        max_batch_size=1
    )
    #app()