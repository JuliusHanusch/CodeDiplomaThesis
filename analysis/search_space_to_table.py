from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import pcs
from ConfigSpace.util import get_one_exchange_neighbourhood
from tabulate import tabulate
import pandas as pd
import os 
import importlib
from pathlib import Path
import sys

def cs_to_latex_table(cs: ConfigurationSpace, filepath: str, caption="", label="", overwrite={}):
    hp_types = {
        "UniformInteger": "Int",
        "UniformFloat": "Float",
        "Categorical": "Cat",
    }
    bool_to_mark = {
        0: r"\xmark",
        1: r"\cmark",
        -1: r"--"
    }
    # Replace too long entries with footnotes in caption
    asterics = [r"$\dagger$", r"$\ddagger$", r"$\S$", r"$\P$"]
    asterics_counter = 0
    footnotes = {}

    cs_dict = dict(cs)
    # Drop constant hyperparameters
    for hp in list(cs.values()):
        if cs[hp.name].get_num_neighbors() <= 1:
            cs_dict.pop(hp.name)

    # Extract parameter data
    rows = []
    for hp_name, hp in cs_dict.items():
        hp_type = type(hp).__name__.replace("Hyperparameter", "")
        if hasattr(hp, 'choices'):
            values = "{" + ", ".join(map(str, hp.choices)) + "}"
        elif hasattr(hp, 'lower') and hasattr(hp, 'upper'):
            values = f"[{hp.lower}, {hp.upper}]"
        else:
            values = "-"
        default = hp.default_value
        log = int(getattr(hp, "log", -1))
        if len(values) > 30 or hp_name in overwrite:
            # Too long for table --> Add in caption instead 
            footnotes[asterics[asterics_counter]] = values if hp_name not in overwrite else overwrite[hp_name]
            values = asterics[asterics_counter]
            asterics_counter += 1
        rows.append([hp.name, hp_types[hp_type], values, default, bool_to_mark[log]])

    df = pd.DataFrame(rows, columns=["Name", "Type", "Domain", "Default", "Log"])

    # Generate LaTeX using booktabs
    latex_table = tabulate(df.values, headers=df.columns, tablefmt="latex_booktabs", showindex=False)
    latex_table = latex_table.replace(r"\textbackslash{}", "\\")
    latex_table = latex_table.replace(r"\$", "$")
    footnotes_strings = [f"{aster} stands for {footnote}".replace('_', '\\_') for aster, footnote in footnotes.items()]
    if len(footnotes_strings) > 0:
        caption = caption + f"{'Some domains were' if len(footnotes_strings) > 1 else 'A domain was'} abbreviated for readability reasons.\n" + ". ".join(footnotes_strings)
    latex_table = latex_table + f"\n\\caption{{{caption}}}\n\\label{{tab:{label}}}"
    latex_table = "\\begin{table}[h]\n" \
                  "\\centering\n" + f"{latex_table}\n" \
                  "\\end{table}\n"
    

    with open(filepath, "w") as f:
        f.write(latex_table)


def load_configspace_from_file(file_path: str, func_name: str):
    path = Path(file_path)
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cs = load_configspace_from_file("../src/search_space.py", func_name="get_config_space")("[]")
    cs_to_latex_table(
        cs=cs, 
        filepath="../cache/tex/search_space.tex", 
        caption="""
        Search space used during HPO. 
        Additionally we also include several versions of our preprocessed corpora, as described in Sect. \\ref{sec:prepro}.
        To each of them a weight between 0 and 1 is assigned (Default is 1 to all, i.e., equal weights).
        Then the resulting vector was normalized such that in sum all their weights add up to 1.
        """,
        label="hpo_search_space")
    
    overwrites = {
        "corpora": "all possible subsets that can be generated from the 7 corpora described before."
    }
    
    cs = load_configspace_from_file("../data/create_data_configs.py", func_name="build_search_space")(42)
    cs_to_latex_table(
        cs=cs, 
        filepath="../cache/tex/search_space_prepro.tex", 
        caption="""
        Search space used for data preparation. 
        """,
        label="prepro_search_space",
        overwrite=overwrites)