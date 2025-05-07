import datetime
from pathlib import Path
import json



def make_dict_storable(advanced_dictionary: dict)->dict:
    """
    Takes in a dict with advanced values like Datatime, dicts, lists, etc. 
    and converts it into a dict that can be stored in an sqlite database meaning strings, numbers, etc.

    Args:
        advanced_dictionary (dict): dict with potentionally complex datatypes

    Returns:
        dict: A dictionary with only simple datatypes
    """
    simple_dict = {}
    for key, value in advanced_dictionary.items():
        if isinstance(value, (int, float, str)):
            pass
        elif isinstance(value, bool):
            value = 1 if value else 0
        elif isinstance(value, (bytes, Path, list)) or value is None:
            value = str(value)
        elif isinstance(value, (datetime.datetime, datetime.date)):
            value = value.strftime("%d-%m-%Y %H:%M:%S")
        elif isinstance(value, dict):
            value = json.dumps(value)
        #elif isinstance(value, str):
        else:
            raise NotImplementedError(f"dtype {type(value)} is currently not storable, but can likely be easily added in make_dict_storable()")
        simple_dict[key] = value

    return simple_dict


def results_to_metrics(results_dict: dict):
    """
    Takes in a the results dict containing two df for in-domain and zero shot eval, extracts the metrics,
    returns all metrics as vaiables
    """
    metrics = ["MASE", "WQL", "MAE", "NRMSE"]
    
    # Ensure consistent keys
    in_domain_df = results_dict.get("in-domain")
    zero_shot_df = results_dict.get("zero-shot")

    # Compute means for each metric
    in_domain_means = {f"in_domain_{metric.lower()}": in_domain_df[metric].mean() for metric in metrics}
    zero_shot_means = {f"zero_shot_{metric.lower()}": zero_shot_df[metric].mean() for metric in metrics}

    # Combine and return as separate variables
    return (
        in_domain_means["in_domain_mase"],
        in_domain_means["in_domain_wql"],
        in_domain_means["in_domain_mae"],
        in_domain_means["in_domain_nrmse"],
        zero_shot_means["zero_shot_mase"],
        zero_shot_means["zero_shot_wql"],
        zero_shot_means["zero_shot_mae"],
        zero_shot_means["zero_shot_nrmse"]
    )

