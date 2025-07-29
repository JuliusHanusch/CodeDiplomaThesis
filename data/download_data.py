import sys  
import os  
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))  
from src.utils import is_in_collection

import datasets as ds
import yaml
import os
from pathlib import Path
from typer import Typer, Option
import subprocess
from shutil import which
import tempfile
from shutil import rmtree

app = Typer(pretty_exceptions_enable=False)


@app.command()
def download_corpus(corpus_name: str = Option(..., help="Name of the corpus to download. Options: 'kaggle', 'chronos', 'lotsa'")):
    """
    Download the specified time series corpus.
    """
    # load test set names
    with open(Path("../chronos_pkg/scripts/evaluation/configs/zero-shot.yaml").resolve(), "r") as f:
        test_set = yaml.safe_load(f)
        # Convert To List + Add Human matches
        test_set_names = [cfg["name"] for cfg in test_set] + ["m4-forecasting-competition-dataset", "monash-oiklab-weather"]

    if corpus_name == "kaggle":
        download_kaggle_corpus(test_set_names=test_set_names)
    elif corpus_name == "uci":
        download_uci_corpus(test_set_names=test_set_names)
    elif corpus_name == "chronos":
        download_chronos_corpus(test_set_names=test_set_names)
    elif corpus_name == "lotsa":
        download_lotsa_corpus(test_set_names=test_set_names)
    elif corpus_name == "all":
        # Download all Sequentially
        download_kaggle_corpus(test_set_names=test_set_names)
        download_chronos_corpus(test_set_names=test_set_names)
        download_uci_corpus(test_set_names=test_set_names)
        download_lotsa_corpus(test_set_names=test_set_names)
    else:
        print("Invalid corpus name. Please choose from 'kaggle', 'chronos', or 'lotsa'.")


def download_kaggle_corpus(test_set_names):
    if which("kaggle") is None:
        print("kaggle missing. pip install kaggle to proceed.")
        return
    # Download Data but dont HF-Cache TS-Corpus has its own caching mechanism
    with tempfile.TemporaryDirectory(dir=cache_path) as tmp_cache_dir:
        kaggle_dataset = ds.load_dataset("ddrg/kaggle-time-series-datasets", "TIME_SERIES", trust_remote_code = True, cache_dir=tmp_cache_dir)
        # Filter out potential matches from test set
        print(f"Corpus before filtering: {str(kaggle_dataset)}")
        kaggle_dataset = kaggle_dataset.filter(lambda x: not is_in_collection(x["name"], test_set_names), num_proc=8)
        print(f"Corpus after filtering: {str(kaggle_dataset)}")
        kaggle_dataset.save_to_disk(output_path/"Time_Corpus")

        
def download_uci_corpus(test_set_names):
    # Download Data but dont HF-Cache TS-Corpus has its own caching mechanism
    with tempfile.TemporaryDirectory(dir=cache_path) as tmp_cache_dir:
        dataset = ds.load_dataset("ddrg/time-series-datasets", "TIME_SERIES", trust_remote_code = True, cache_dir=tmp_cache_dir)
        dataset = dataset.filter(lambda x: not is_in_collection(x["name"], test_set_names), num_proc=8)
        dataset.save_to_disk(output_path/"UCI_Corpus")


def download_chronos_corpus(test_set_names):
    for super_dataset in ["autogluon/chronos_datasets", "autogluon/chronos_datasets_extra"]:
        if "extra" in super_dataset: # Needs to be set manually 
            subdataset_names = ["ETTh", "ETTm", "spanish_energy_and_weather", "brazilian_cities_temperature"]
        else:
            subdataset_names = ds.get_dataset_config_names(super_dataset, trust_remote_code=True)
        for subdataset_name in subdataset_names:
            sub_ds = ds.load_dataset(super_dataset, subdataset_name, trust_remote_code=True, cache_dir=cache_path)
            if subdataset_name in test_set_names:
                sub_ds.save_to_disk(output_path/f"Chronos_Corpus_ZEROSHOT/{subdataset_name}")
            elif "kernel_synth" in subdataset_name: # Make it Kernel Synth Extra Corpus
                sub_ds.save_to_disk(output_path/f"Chronos_Corpus_Kernel_Synth/{subdataset_name}")
            elif "tsmixup" in subdataset_name: # Remove TS MIxup because already augmented
                pass
            else:
                sub_ds.save_to_disk(output_path/f"Chronos_Corpus/{subdataset_name}")


def download_lotsa_corpus(test_set_names):
    repo = "Salesforce/lotsa_data"
    local_dir = str(output_path/"Lotsa_Corpus")

    # Download dataset
    try:
        subprocess.run([
            "huggingface-cli", "download", repo,
            "--repo-type=dataset",
            "--local-dir", local_dir
        ], check=True)
    except:
        print("Download Broke but maybe CLI Download Succeded?!")
    # Delete all DS in the test set from it 
    # loop over all ds in folder
    local_dir = Path(local_dir)
    downloaded_datasets = [(dir.name, dir) for dir in local_dir.iterdir() if dir.is_dir() and dir.name[0] != "."]
    for dataset_name, dataset_folder in downloaded_datasets:
        if is_in_collection(dataset_name, test_set_names):
            # deleted dataset again to not accidentally train on test set
            rmtree(dataset_folder)
            print(f"Removed {dataset_folder}")     


if __name__ == "__main__":
    # Go to current file directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Read Credentials
    credentials_path = Path("../credentials.yml")
    cache_path = Path("./cache")
    output_path = Path("./data_sets_raw")
    cache_path.mkdir(parents=True, exist_ok=True)

    # Load the config file
    try:
        with open(credentials_path, "r") as f:
            config = yaml.safe_load(f)
    except:
        print("Missing credentials.yml file. Run generate_credentials_file.py")
        exit(1)
    os.environ["KAGGLE_USERNAME"] = config["KAGGLE"]["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = config["KAGGLE"]["KAGGLE_KEY"]
    os.environ["HF_TOKEN"] = config["HUGGINGFACE"]["HF_TOKEN"]
    os.environ["HF_DATASETS_DISABLE_CACHING"] = "1"

    # Download Specific Corpus
    app()


    
    

