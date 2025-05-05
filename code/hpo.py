# Include Parent Directory to load packages from
import sys  
import os  
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))  
from chronos.scripts.training.train import main as train_chronos
from chronos.scripts.evaluation.evaluate import main as eval_chronos
from ConfigSpace import Configuration, ConfigurationSpace
from functools import partial
from pathlib import Path


BASE_OUTPATH = Path("./chronos_models")

def train(config: Configuration, training_data_paths: str, seed: int = 0, budget: int = 0):
    training_steps = budget *  1000 # 30k 90k 270k 810k 2430k -> Best Config Trained for 2.5M steps

    # Hash Config to get exactly one model per config
    config_dict = dict(config)
    config_dict["seed"] = seed
    config_dict["training_steps"] = training_steps
    config_hash = hash(frozenset(config_dict.items()))

    output_path =  Path(BASE_OUTPATH / f"chronos_{config_hash}")

    if output_path.exists():
        raise NotImplementedError(f"Model already trained for config {config_hash}.")
        # TODO Load Config Results from DB and return
        return

    # ! Train Chronos
    # TODO Pass all HPs to Chronos
    train_chronos(
        output_dir=output_path,
        training_data_paths=training_data_paths,
        max_steps=training_steps,
    )

    # ! Eval Chronos
    eval_chronos(
        # TODO config_path = Valset
        chronos_model_id = output_path,
        
    )


    # ! Save Meta Data To DB

    # ! Return Costs