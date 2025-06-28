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


BASE_OUTPATH = Path("./chronos_models")

app = Typer()

@app.command()
@use_yaml_config(param_name="config")
def main(
    seed: int = Option(0, help="Random seed for reproducibility."),
    model_id: str = Option("google/t5-efficient-tiny", help="Which base model to use."),
    trial_walltime_limit: int = Option(300, help="How long until we stop a trial. (-1 ~ Unlimited)"),
    number_trials: int = Option(5, help="How many trials to run."),
    min_budget: int = Option(30_000, help="Minimum number of Training Steps"),
    max_budget: int = Option(2_500_000, help="Maximum number of Training Steps"),
    eta: int = Option(3, help="Hyperband eta."),
    memory: str = Option("160G", help="Memory to allocate for each worker."),
    worker_walltime: str = Option("01:00:00", help="How long shall each worker live."),
    account: str = Option("chronos_project", help="To which project shall the jobs be assigned. none for no project."),
    job_extra_directives: str = Option("['--gres=gpu:1']", help="Extra directives to be passed to the job scheduler."),
    worker_count: int = Option(1, help="How many workers to use."),
):

    configs_space = get_config_space(model_id=model_id)


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
        objectives=["WQL_ZS", "MASE_ZS", "WQL_ID", "MASE_ID"],  # TODO Add NRMSE
    )

    # TODO
    cluster = SLURMCluster(
        cores=1, #TODO Can we increase to 4
        memory=memory,
        walltime=worker_walltime,
        job_extra_directives=literal_eval(job_extra_directives),
        account=account if account.lower() != "none" else None,
        job_name="chronos_hpo",
        local_directory=str((root_dir / "hpc/logs/dask").resolve()),
        processes=1,
        log_directory=str((root_dir / "hpc/logs/smac").resolve()),
        nanny = False,
        job_script_prologue=[
            f"cd {root_dir.resolve()}",
            "pwd",
            "source ./hpc/modules.sh",
        ],
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
            objective_weights=[2, 2, 1, 1],  # TODO Weights Zeroshot twice as much as in domain (To avoid overweighting datasets in validation set)
        ),
    )

    sleep(35)

    # Let's optimize
    incumbents = smac.optimize()


    for incumbent in incumbents:
        print(f"Best configuration: {incumbent}")
        cost = smac.validate(incumbent)
        print("---", cost)

    client.close()
    cluster.close()




def train(
    config: Configuration, 
    seed: int = 0, 
    budget: int = 1
    ):
    training_steps = int(budget) # 30k 90k 270k 810k 2430k -> Best Config Trained for 2.5M steps

    # Hash Config to get exactly one model per config
    config_dict = dict(config)
    config_dict["seed"] = seed
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
    print(f"Tokenizer Kwargs: {tokenizer_kwargs}")

    # Train Chronos and Save to outpath
    model_path = trainer.main(
        
        output_dir=output_path,
        training_data_paths=training_data_paths,
        max_steps=training_steps,
        tokenizer_kwargs=tokenizer_kwargs,
        d_kv=d_kv,
        d_ff=d_ff,
        **config, 
    )


    # ! Eval Chronos
    val_configs = (
        #Path(literal_eval(training_data_paths)[0]),
        Path("./data/eval_configs/eval_config.yml").resolve(), # TODO Path to Zero-Shot Val Config
        #Path("./Chronos/Scripts/evaluation/configs/zero-shot.yaml"), # TODO Path to In-Domain Val Config
    )
    results = {}

    print(f"Start Evaluating {model_path} on {val_configs}")
    for val_config in val_configs:
        results[val_config.stem] = evaluater.main(
            config_path=val_config,
            chronos_model_id = model_path,
            metrics_path="./cache/validation_scores.csv",
            device="cuda",
            torch_dtype="bfloat16",
            batch_size=32,
            num_samples=20,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        )

    # TODO Check that all metrics are near 0-1 range (e.g. NRMSE)
    import pandas as pd
    pd.options.display.max_columns = None
    print(f"Results:\n {results}")

    # ! Save Meta Data To DB

    from src.db import insertTable

    config_simple = make_dict_storable(config_dict)
    in_domain_mase, in_domain_wql, in_domain_mae, in_domain_nrmse, zero_shot_mase, zero_shot_wql, zero_shot_mae, zero_shot_nrmse = results_to_metrics(results)
    insertTable("Results", {"id":config_hash, "config":config_simple, "ModelPath":output_path,
                            "in_domain_mase":in_domain_mase, "in_domain_wql":in_domain_wql, "in_domain_mae":in_domain_mae, "in_domain_nrmse":in_domain_nrmse,
                            "zero_shot_mase":zero_shot_mase,"zero_shot_wql":zero_shot_wql,"zero_shot_mae":zero_shot_mae,"zero_shot_nrmse":zero_shot_nrmse })

    # ! Return Costs
    return # TODO ["WQL_ZS", "MASE_ZS", "WQL_ID", "MASE_ID"]



if __name__ == "__main__":
    # Go to current file directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


    # TODO
    main(
        config={}, # TODO Add Config
        seed=0,
        model_id="google/t5-efficient-tiny",
        trial_walltime_limit=-1,
        number_trials=5,
        min_budget=100,
        max_budget=300,
        eta=3,
        memory="160G",
        worker_walltime="01:00:00",
        account="p_automl",
        job_extra_directives="['--gres=gpu:1']",
        worker_count=2,
    )
    #app()