from ucimlrepo import fetch_ucirepo 
from pathlib import Path 
import yaml
import logging
import typer
from gluonts.dataset.split import split
from evaluate import to_gluonts_univariate
import datasets
from datasets import Dataset
import pandas as pd
from datasets import Features, Value
import zipfile
import tempfile
import subprocess
import shutil
from ucimlrepo import DatasetNotFoundError





app = typer.Typer()

@app.command()
def main(
    config_path: Path,
):
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    for config in backtest_configs:
        print(f"\n=== Processing dataset: {config.get('name', config.get('id', 'unknown'))} ===")
        print(f"\nStarting processing: {config['name']}")

        if "hf_repo" in config:
            offset = config["offset"]
            prediction_length = config["prediction_length"]
            num_rolls = config["num_rolls"]
            hf_repo = config["hf_repo"]
            name = config["name"]
            trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False


            ds = datasets.load_dataset(
                hf_repo, name, split="train", trust_remote_code=trust_remote_code
            )

            ds.set_format("numpy")

            gts_dataset = to_gluonts_univariate(ds)

            # Split dataset for evaluation
            _, test_template = split(gts_dataset, offset=offset)
            validation_data = test_template.generate_instances(prediction_length, windows=num_rolls)

            print("Finished dataset")


        else:
            dataset_id = config["id"]
            targets = config["targets"]
            offset = config["offset"]
            prediction_length = config["prediction_length"]
            num_rolls = config["num_rolls"]

            try:
                ds = fetch_ucirepo(id=dataset_id)
                print("Dataset can be fetched directly")
                print(dataset_id)

                df = pd.concat([ds.data.features, ds.data.targets], axis=1)

                for target in targets:
                    try:
                        if target not in df.columns:
                            print(f"Skipping missing target '{target}' in dataset ID '{dataset_id}'")
                            continue

                        data = df[[target]].copy()
                        data.reset_index(inplace=True)

                        # Clean all values (remove $ and , from strings)
                        def clean_currency(val):
                            if isinstance(val, str):
                                return pd.to_numeric(val.replace("$", "").replace(",", ""), errors="coerce")
                            return val

                        data = data.applymap(clean_currency)

                        features = Features({col: Value("float64") for col in data.columns})

                        dataset = Dataset.from_pandas(data, features=features)

                        _, test_template = split(dataset, offset=offset)
                        validation_data = test_template.generate_instances(prediction_length, windows=num_rolls)

                    except Exception as e:
                        print(f"Skipping target '{target}' in dataset ID '{dataset_id}' due to error: {e}")
                        continue

                print("Finished dataset")

            except DatasetNotFoundError:
                try:
                    url = config["link"]
                    filename = config.get("filename")
                    dataset_name = config["name"]

                    local_zip = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
                    print("Downloading zip...")
                    subprocess.run(["wget", "-q", "-O", local_zip.name, url], check=True)
                    print("Downloaded zip to", local_zip.name)

                    print("Extracting main zip...")
                    extract_dir = Path(tempfile.mkdtemp())
                    with zipfile.ZipFile(local_zip.name, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    print("Main extraction done.")

                    def extract_nested_zips(dir_path):
                        nested_zips = list(dir_path.glob("**/*.zip"))
                        for nested_zip in nested_zips:
                            with zipfile.ZipFile(nested_zip, "r") as zip_ref:
                                zip_ref.extractall(nested_zip.parent)
                            nested_zip.unlink()
                        if list(dir_path.glob("**/*.zip")):
                            extract_nested_zips(dir_path)

                    extract_nested_zips(extract_dir)

                    filename = config.get("filename")
                    if filename is None:
                        raise ValueError(f"'filename' must be specified in config for dataset '{dataset_name}'")

                    target_path = extract_dir / Path(filename)
                    print(f"Looking for file: {target_path}")
                    if not target_path.exists():
                        print("Extracted files:")
                        for path in extract_dir.rglob("*"):
                            print(f" - {path.relative_to(extract_dir)}")
                        raise FileNotFoundError(f"'{filename}' not found in extracted contents of dataset '{dataset_name}'")

                    if target_path.suffix.lower() in [".csv", ".txt"]:
                        print("Reading CSV...")
                        df = pd.read_csv(target_path, on_bad_lines='skip')  # pandas >= 1.3
                        print("Finished reading, shape:", df.shape)
                    else:
                        raise ValueError(f"Unsupported file type: {target_path.suffix}")

                    for target in targets:
                        try:
                            if target not in df.columns:
                                print(f"Skipping missing target '{target}' in dataset '{dataset_name}'")
                                continue

                            data = df[[target]].copy()
                            data.reset_index(inplace=True)

                            def clean_currency(val):
                                if isinstance(val, str):
                                    return pd.to_numeric(val.replace("$", "").replace(",", ""), errors="coerce")
                                return val

                            data = data.applymap(clean_currency)

                            features = Features({col: Value("float32") for col in data.columns})
                            dataset = Dataset.from_pandas(data, features=features)

                            _, test_template = split(dataset, offset=offset)
                            validation_data = test_template.generate_instances(prediction_length, windows=num_rolls)

                        except Exception as e:
                            print(f"Skipping target '{target}' in dataset '{dataset_name}' due to error: {e}")
                            continue

                    print(f"Finished dataset: {dataset_name}")

                except Exception as e:
                    print(f"Failed to process dataset '{config['name']}': {e}")

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("UCI Data")
    logger.setLevel(logging.INFO)
    app()
