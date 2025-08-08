import yaml
from typer import Typer
import typer
from typing import Optional
from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float
import numbers

app = Typer()

def all_subsets(collection: list):
    if len(collection) == 0:
        return [collection]
    else:
        components = all_subsets(collection=collection[1:])
        return [collection[0:1] + component for component in components] + components

def build_search_space(seed: Optional[int] = None) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)

    corpora = [ 
        # Our Corpora
        "Time_Corpus_Processed",
        "Time_Corpus_Processed_Split",
        "UCI_Corpus_Processed",
        "UCI_Corpus_Processed_Split",
        # Old Corpora
        "Lotsa_Corpus",
        "Chronos_Corpus",
        "Chronos_Corpus_Kernel_Synth",
    ]
    corpora = all_subsets(corpora)
    corpora = [corpus for corpus in corpora if corpus] # Remove empty list

    cs.add(
        Categorical(
            name="corpora",
            items=corpora,
            default=["Chronos_Corpus"]
        )
    )
    cs.add(Integer("k", (1, 6), default=3))
    cs.add(Integer("length_expo", (7, 12), default=9))
    cs.add(Float("alpha", (1e-3, 25.0), log=True, default=1))
    cs.add(Float("small_ts_share", (0.001, 2), log=True, default=0.1))
    cs.add(Categorical("deduplication", [1, 0], default=0))

    return cs


@app.command()
def main(
    count: int = typer.Option(5, "--count", "-n", help="Number of configs to generate"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
):
    """
    Sample `count` many random configurations from the defined search space
    and print each as YAML.
    """
    cs = build_search_space(seed)
    configs = [dict(cs.get_default_configuration())] + [dict(config) for config in cs.sample_configuration(count-1)]
    
    print(configs)
    for i, config in enumerate(configs):
        config["length"] = 2 ** config.pop("length_expo")
        config = {key: val if not isinstance(val, numbers.Number) else float(val) for key, val in config.items()}
        with open(f"./data/data_configs/{i}.yml", "w") as file:
            yaml.safe_dump(config, file)


if __name__ == "__main__":
    app()
