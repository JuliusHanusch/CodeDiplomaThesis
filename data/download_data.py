import datasets as ds
import yaml
import os
from pathlib import Path
from typer import Typer, Option
import subprocess

app = Typer()

@app.command()
def download_corpus(corpus_name: str = Option(..., help="Name of the corpus to download. Options: 'kaggle', 'chronos', 'lotsa'")):
    """
    Download the specified time series corpus.
    """
    if corpus_name == "kaggle":
        download_kaggle_corpus()
    elif corpus_name == "chronos":
        download_chronos_corpus()
    elif corpus_name == "lotsa":
        download_lotsa_corpus()
    elif corpus_name == "all":
        # Download all Sequentially
        download_kaggle_corpus()
        download_chronos_corpus()
        download_lotsa_corpus()
    else:
        print("Invalid corpus name. Please choose from 'kaggle', 'chronos', or 'lotsa'.")


def download_kaggle_corpus():
    kaggle_dataset = ds.load_dataset("ddrg/kaggle-time-series-datasets", "TIME_SERIES", trust_remote_code = True, cache_dir=cache_path)
    kaggle_dataset.save_to_disk(output_path/"Time_Corpus")

def download_chronos_corpus():
    subdatasets = []
    for super_dataset in ["autogluon/chronos_datasets", "autogluon/chronos_datasets_extra"]:
        subdataset_names = ds.get_dataset_config_names(super_dataset, trust_remote_code=True)
        for subdataset_name in subdataset_names:
            sub_ds = ds.load_dataset(super_dataset, subdataset_name, trust_remote_code=True, cache_dir=cache_path)
            sub_ds.save_to_disk(output_path/f"Chronos_Corpus/{subdataset_name}")
            #subdatasets.append(sub_ds)
        
    #chronos_dataset = ds.concatenate_datasets(subdatasets)
    #chronos_dataset.save_to_disk(output_path/"Chronos_Corpus")

def download_lotsa_corpus():
    repo = "Salesforce/lotsa_data"
    local_dir = str(output_path/"Lotsa_Corpus")

    # Download dataset
    subprocess.run([
        "huggingface-cli", "download", repo,
        "--repo-type=dataset",
        "--local-dir", local_dir
    ], check=True)



if __name__ == "__main__":
    # Go to current file directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Read Credentials
    credentials_path = Path("../credentials.yml")
    cache_path = Path("./cache")
    output_path = Path("./data_sets_raw")
    cache_path.mkdir(parents=True, exist_ok=True)

    # Load the config file
    with open(credentials_path, "r") as f:
        config = yaml.safe_load(f)

    os.environ["KAGGLE_USERNAME"] = config["KAGGLE"]["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = config["KAGGLE"]["KAGGLE_KEY"]
    os.environ["HF_TOKEN"] = config["HUGGINGFACE"]["HF_TOKEN"]
    os.environ["HF_DATASETS_DISABLE_CACHING"] = "1"

    # Download Specific Corpus
    app()


    
    

