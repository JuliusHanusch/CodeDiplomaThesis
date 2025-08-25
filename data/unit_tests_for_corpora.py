import pandas as pd
from pathlib import Path
import datasets as ds
import yaml
import os
from huggingface_hub import hf_hub_download
from collections import Counter
from warnings import warn 
import sys

data_directory = Path("./data_sets_raw")

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    with open("../credentials.yml") as f:
        credentials = yaml.safe_load(f)

    # ======= #
    # Chronos #
    # ======= #
    chr_base = data_directory / "Chronos_Corpus"
    chr_kernel_synth = data_directory / "Chronos_Corpus_Kernel_Synth"
    chr_test = data_directory / "Chronos_Corpus_ZEROSHOT"
    chr_corpora = [chr_base, chr_kernel_synth, chr_test]
    dataset_paths = [p for corpus in chr_corpora for p in corpus.iterdir() if p.is_dir()]
    ds_names = [p.stem for p in dataset_paths]


    # Check: All Expected DS Included 
    expected_ds = ds.get_dataset_config_names("autogluon/chronos_datasets", trust_remote_code=True)
    expected_ds += ["ETTh", "ETTm", "spanish_energy_and_weather", "brazilian_cities_temperature"]
    for ds_name in expected_ds:
        if ds_name in ds_names:
            assert ds_name != "training_corpus_ts_mixup_10m", "Chronoses TSMixup corpus shouldn't be here, as we''ll create our own"
            continue # AOK
        elif ds_name == "training_corpus_tsmixup_10m":
            continue # The one Exceptions
        else: 
            raise Exception(f"{ds_name} is missing in Chronos Corpus, to achieve reproducibility delete caches and try download it again")
    print("Chronos Corpus is Complete")

    
    # Check: No Zeroshot Test set in the chronos train corpora
    with open(Path("../chronos_pkg/scripts/evaluation/configs/zero-shot.yaml").resolve(), "r") as f:
        test_set = yaml.safe_load(f)
        # Convert To List + Add Human matches
        test_set_names = [cfg["name"] for cfg in test_set] + ["m4-forecasting-competition-dataset", "monash-oiklab-weather"]

    chr_train = [p.stem for corpus in [chr_base, chr_kernel_synth] for p in corpus.iterdir() if p.is_dir()]
    overlap = set(chr_train).intersection(test_set_names)
    assert len(overlap) == 0, f"Test Set leaked into train data, {overlap} is/are in train corpora"

    # Each ds can be loaded and has over 100 rows
    for p in dataset_paths:
        dataset = ds.load_from_disk(p)["train"]
        if p.stem in ("ercot", "exchange_rate", "monash_pedestrian_counts", "monash_cif_2016", "monash_australian_electricity", "brazilian_cities_temperature"):
            # A few exceptions that are expected to be shorter
            assert len(dataset) >= 5
            continue 
        elif p.stem in ("monash_saugeenday", "spanish_energy_and_weather", "ETTm", "ETTh"):
            assert len(dataset) >= 1
        else:
            assert len(dataset) > 100, f"Dataset {p.stem} is too short with {len(dataset)} something might have gone wrong during download"
    print(f"All Chronos Datasets of sufficient size.")
    

    # Check Disjunction
    for corpus1 in chr_corpora:
        for corpus2 in chr_corpora:
            if corpus1 == corpus2:
                continue
            else:
                ds_names1 = [p.stem for p in corpus1.iterdir() if p.is_dir()]
                ds_names2 = [p.stem for p in corpus2.iterdir() if p.is_dir()]
                overlap = set(ds_names1).intersection(ds_names2)
                assert len(overlap) == 0, f"{overlap} are in both {corpus1} and {corpus2}"
    print("No overlap btwn. Chronos Corpora detected")


    # ============ # 
    # LOTSA Corpus #
    # ============ # 

    # As long as specified in the paper
    lotsa_folder = data_directory / "Lotsa_Corpus"
    subfolders = [f for f in lotsa_folder.iterdir() if not f.name.startswith('.') and f.is_dir()]
    print(len(subfolders))

    # Check if all DS downloaded
    # 174 originally, 12 we removed as they are included in the test set, 4 use share folder with other ds e.g. cmip6, default gets dropped
    assert len(subfolders) == 158, f"Only {len(subfolders)} of 158 expected DS in LOTSA Corpus, try to download it again"
    extra_short_ds = ( # We know that those are shorter than the norm
        "australian_electricity_demand", "oikolab_weather", "saugeenday", 
        "pdb", "smart", "gfc17_load", "covid19_energy", "cockatoo", 
        "elecdemand", "us_births", "solar_power", "elf", "wind_power", 
        "spain", "sceaux", "sunspot_with_missing", "gfc14_load"
        )
    print("All expected LOTSA DS are here")

    # Check if all have sufficient size
    for subfolder in subfolders:
        dataset = ds.load_from_disk(subfolder)
        ds_length =len(dataset)
        if ds_length < 10:
            if subfolder.stem in extra_short_ds: 
                # AOK Those have excemption - They are naturally short
                continue
            else:
                raise Exception(f"Dataset in {subfolder} is suspiciously short with only {ds_length} rows")
    print("All LOTSA DS surpass there minimum size requirement")

    # ============ #
    # Time Corpora #
    # ============ #

    def verify_corpus_integrity(corpus_name: str, 
                            index_file: str, 
                            repo_id: str, 
                            data_dir: Path,
                            token: str,
                            allowed_missing_pct: float):
        # Load dataset index
        index_path = hf_hub_download(
            repo_id=repo_id,
            filename=index_file,
            repo_type="dataset",
            token=token,
        )
        index_df = pd.read_csv(index_path, delimiter=';')
        if "url" in index_df.columns:
            index_df = index_df.drop_duplicates(subset=['url', 'file_name'], keep='first')
        else:
            index_df = index_df.drop_duplicates(subset=['datasetID', 'file_name'], keep='first')
        # Explode 
        index_df["data_column"] = index_df["data_column"].apply(lambda x: x.split(","))
        index_df = index_df.explode("data_column", ignore_index=True)
        # Count Expected Entries
        expected = Counter(index_df["name"])

        # Load full corpus and flatten
        full_corpus = ds.load_from_disk(data_dir / corpus_name)
        if isinstance(full_corpus, dict):  # has train/test splits
            full_corpus = ds.concatenate_datasets([full_corpus["train"], full_corpus["test"]])
        full_corpus = full_corpus.to_pandas()
        full_corpus = full_corpus.explode("value", ignore_index=True)
        observed = Counter(full_corpus["name"])

        # Compute missing
        deficit = expected - observed
        missing_pct = sum(deficit.values()) / len(index_df)

        for key, value in deficit.items():
            warn(f"From {key} are {value}/{expected[key]} TS missing")
        assert missing_pct < allowed_missing_pct, f"{missing_pct*100:.2f}% of {corpus_name} DS are missing, reproducibility might be impaired"
        print(f"Missing {missing_pct*100:.2f}% of {corpus_name} DS - That is rather normal")

    
    verify_corpus_integrity(
        corpus_name="Time_Corpus",
        index_file="time-series-datasets.csv",
        repo_id="ddrg/kaggle-time-series-datasets",
        data_dir=data_directory,
        token=credentials["HUGGINGFACE"]["HF_TOKEN"],
        allowed_missing_pct=0.1125, #We only achieved about 89% available data (A few were removed on purpose as they might be in test set many might get lost when DS change on kaggle)
    )

    verify_corpus_integrity(
        corpus_name="UCI_Corpus",
        index_file="UCI_final.csv",
        repo_id="ddrg/time-series-datasets",
        data_dir=data_directory,
        token=credentials["HUGGINGFACE"]["HF_TOKEN"],
        allowed_missing_pct=0.023 
    )
