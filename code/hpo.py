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
import logging

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
from search_space import get_config_space
from smac import HyperparameterOptimizationFacade as HPOFacade
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from time import sleep

BASE_OUTPATH = Path("./chronos_models")

app = Typer()

@app.command()
def main(
    config: Configuration = Option(..., help="Configuration for the model."),
    training_data_paths: str = Option(..., help="Path to the training data."),
    seed: int = Option(0, help="Random seed for reproducibility."),
    model_id: str = Option("google/t5-efficient-tiny", help="Which base model to use."),
):

    configs_space = get_config_space(model_id=model_id)


    # Define our environment variables
    scenario = Scenario(
        configspace=configs_space,
        trial_walltime_limit=600,  # TODO After 60 seconds, we stop the hyperparameter optimization
        n_trials=10,  # TODO Evaluate max 500 different trials
        min_budget=1,  # TODO Train the MLP using a hyperparameter configuration for at least 5 epochs
        max_budget=25,  # TODO Train the MLP using a hyperparameter configuration for at most 25 epochs
        # n_workers=8, not by cluster jobs
        seed=seed,
        objectives=["WQL_ZS", "MASE_ZS", "WQL_ID", "MASE_ID"],  # TODO Add NRMSE
    )

    # TODO
    cluster = SLURMCluster(
        cores=1,
        memory="160G",
        walltime="01:00:00",
        job_extra_directives=["--gres=gpu:1"],
        account="p_automl",
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

    cluster.scale(jobs=2)  # Ask for 1 job

    client = Client(address=cluster)

    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

    # Create our intensifier
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget", seed=seed, eta=3)

    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario=scenario,
        target_function=partial(train, training_data_paths=training_data_paths), # TODO Set Data Path Algorithmically
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
        dask_client=client,
        multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(
            scenario,
            objective_weights=[2, 2, 1, 1],  # Weights Zeroshot twice as much as in domain (To avoid overweighting datasets in validation set)
        ),
    )

    sleep(10)

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
    training_data_paths: str, 
    seed: int = 0, 
    budget: int = 1
    ):
    training_steps = budget *  1000 # 30k 90k 270k 810k 2430k -> Best Config Trained for 2.5M steps

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
    from code.evaluate import main as eval_chronos
    import code.train as trainer
    # Set Missing Global Variables
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    trainer.logger = logger

    # Special HPs
    tokenizer_limit = config.pop("tokenizer_limit", 15)
    tokenizer_kwargs = f"{{'low_limit': -{tokenizer_limit:.3f}, 'high_limit': {tokenizer_limit:.3f}}}"
    print(f"Tokenizer Kwargs: {tokenizer_kwargs}")

    # Train Chronos and Save to outpath
    trainer.main(
        output_dir=output_path,
        training_data_paths=training_data_paths,
        #max_steps=training_steps,
        tokenizer_kwargs=tokenizer_kwargs,
        **config, 
    )


    # ! Eval Chronos
    val_configs = (
        Path("../Chronos/Scripts/evaluation/configs/in-domain.yaml"), # TODO Path to Zero-Shot Val Config
        Path("../Chronos/Scripts/evaluation/configs/zero-shot.yaml"), # TODO Path to In-Domain Val Config
    )
    results = {}

    for val_config in val_configs:
        results[val_config.stem] = eval_chronos(
            config_path=val_config,
            chronos_model_id = output_path,
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
    print(f"Results: {results}")

    # ! Save Meta Data To DB

    # ! Return Costs
    return # TODO ["WQL_ZS", "MASE_ZS", "WQL_ID", "MASE_ID"]



if __name__ == "__main__":
    from evaluate import main as eval_chronos
    # Go to current file directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


    # TODO
    main(
        config={}, # TODO Add Config
        training_data_paths="['/data/horse/ws/jipo020b-aion/AION/data/testfiles/training_mix.arrow']", # TODO Path to Chronos Data
        seed=0,
    )
    #app()