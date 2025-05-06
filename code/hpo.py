# Include Parent Directory to load packages from
import sys  
import os  
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))  
sys.path.append(str((Path(__file__).parent.parent / "chronos_pkg/src").resolve()))
from chronos_pkg.scripts.training.train import main as train_chronos
from chronos_pkg.scripts.evaluation.evaluate import main as eval_chronos
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

BASE_OUTPATH = Path("./chronos_models")

app = Typer()

@app.command()
def main(
    config: Configuration = Option(..., help="Configuration for the model."),
    training_data_paths: str = Option(..., help="Path to the training data."),
    seed: int = Option(0, help="Random seed for reproducibility."),
    dask_client: str = Option(None, help="Dask cluster address."),
    model_id: str = Option("google/t5-efficient-tiny", help="Which base model to use."),
):

    configs_space = get_config_space(model_id=model_id)


    # Define our environment variables
    scenario = Scenario(
        configspace=configs_space,
        walltime_limit=60,  # TODO After 60 seconds, we stop the hyperparameter optimization
        n_trials=1,  # TODO Evaluate max 500 different trials
        min_budget=1,  # TODO Train the MLP using a hyperparameter configuration for at least 5 epochs
        max_budget=25,  # TODO Train the MLP using a hyperparameter configuration for at most 25 epochs
        n_workers=8,
        seed=seed,
        objectives=["WQL_ZS", "MASE_ZS", "WQL_ID", "MASE_ID"],  
    )

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
        dask_client=dask_client,
        multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(
            scenario,
            objective_weights=[2, 2, 1, 1],  # Weights Zeroshot twice as much as in domain (To avoid overweighting datasets in validation set)
        ),
    )

    # Let's optimize
    incumbents = smac.optimize()


    for incumbent in incumbents:
        print(f"Best configuration: {incumbent}")
        cost = smac.validate(incumbent)
        print("---", cost)




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

    # Import Train Function
    import chronos_pkg.scripts.training.train as trainer
    # Set Missing Global Variables
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    trainer.logger = logger
    # Train Chronos and Save to outpath
    trainer.main(
        output_dir=output_path,
        training_data_paths=training_data_paths,
        #max_steps=training_steps,
        **dict(config), 
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

    print(f"Results: {results}")

    # ! Save Meta Data To DB

    # ! Return Costs
    return # TODO ["WQL_ZS", "MASE_ZS", "WQL_ID", "MASE_ID"]



if __name__ == "__main__":
    # Go to current file directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


    # TODO
    main(
        config={}, # TODO Add Config
        training_data_paths="['/data/horse/ws/jipo020b-aion/AION/data/testfiles/training_mix.arrow']", # TODO Path to Chronos Data
        seed=0,
        dask_client=None,
    )
    #app()