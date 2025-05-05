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
from typer import Typer, Option


BASE_OUTPATH = Path("./chronos_models")

app = Typer()

@app.command()
def main(
    config: Configuration = Option(..., help="Configuration for the model."),
    training_data_paths: str = Option(..., help="Path to the training data."),
    seed: int = Option(0, help="Random seed for reproducibility."),
    budget: int = Option(0, help="Budget for training."),
):
    # TODO Setup Search Space
    # TODO Setup SMAC
    pass



def train(
    config: Configuration, 
    training_data_paths: str, 
    seed: int = 0, 
    budget: int = 0
    ):
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

    # ! Train Chronos and Save to outpath
    train_chronos(
        output_dir=output_path,
        training_data_paths=training_data_paths,
        max_steps=training_steps,
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



if __name__ == "__main__":
    # Go to current file directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    train(
        config={}, # TODO Add Config
        training_data_paths="./data_sets_raw/Chronos_Corpus", # TODO Path to Chronos Data
        seed=0,
        budget=0,
    )
    #app()