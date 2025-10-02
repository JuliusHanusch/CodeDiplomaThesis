"""
Some Mistakes happen during SMAC's checkpointing which lead to bugs and ineffiencies.
Hence, This Script is for Cleaning up Checkpoints such that they can be used more efficiently
"""
from pathlib import Path
import shutil
from datetime import datetime
import json
from ConfigSpace import ConfigurationSpace, Configuration
import os 


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_tracker_length(tracker: dict):
    # For Debug Only
    length = 0
    for stage in tracker.values():
        for tup in stage:
            length += len(tup[1])
    return length


def sanitize(path: Path, seed, min_budget, eta, max_budget, cs: ConfigurationSpace):
    # Duplicate Old Checkpoint
    backup_pth = path.parent / (path.name + "_" + datetime.now().strftime("%Y%m%d_%H%M"))
    shutil.copytree(path, backup_pth)
    # Load all files
    intensifier_path = path/str(seed)/"intensifier.json"
    runhistory_path = path/str(seed)/"runhistory.json"
    intensifier = read_json(intensifier_path)
    runhistory = read_json(runhistory_path)

    broken_configs = []
    meta_data = []
    configs = {}
    origins = {}
    config_id = 1
    tracker = intensifier["state"]["tracker"]
    removed_counter = 0
    for meta in runhistory["data"]:
        old_id = meta["config_id"]
        config = runhistory["configs"][str(old_id)]
        if meta["status"] == 1:
            meta["config_id"] = config_id
            origins[str(config_id)] = runhistory["config_origins"][str(old_id)]
            configs[str(config_id)] = config
            meta_data.append(meta)

            config_id += 1
        else:
            broken_configs.append(config)

            #! Goal: remove all broken configs from the intensifier
            # Problem: It is really hard to say to which bracket & stage a config belongs
            # Solution: Loop over all brackets with same budget -> Remove config from each
            for key, value in tracker.items():
                bracket, stage = key.split(",")
                # TODO: Fix Problem that ACTUAL min budget is calculated by smac (not set by user)
                expected_budget = min_budget * (eta**(int(stage)+int(bracket))) 
                if expected_budget*0.99 >= meta["budget"] or meta["budget"] >= expected_budget*1.01:
                    continue

                # remove config if existent at this budget level
                for seed, cfgs in value:
                    if config in cfgs:
                        # Should remove it in place 
                        cfgs.remove(config)
                        removed_counter += 1
                        print(f"removed {removed_counter} configs from intensifier")

    
    # The intensifier contains additional Configs (which somehow didnt go through the CS Constraints)
    # Check and remove them if necessary
    removed_counter_extra = 0
    for key, value in tracker.items():
        bracket, stage = key.split(",")

        # remove config if existent at this budget level
        for seed, cfgs in value:
            for config in cfgs:
                try:
                    config_ = Configuration(cs, values=config)
                except:
                    # Config is invalid
                    cfgs.remove(config)
                
                    removed_counter_extra += 1
                    print(f"removed {removed_counter}+{removed_counter_extra} configs from intensifier")
                    

    runhistory["data"] = meta_data
    runhistory["configs"] = configs
    runhistory["config_origins"] = origins
    runhistory["stats"] = {
                            "submitted": len(configs),
                            "finished": len(configs),
                            "running": 0
                        }
    intensifier["state"]["tracker"] = tracker
    write_json(runhistory_path, runhistory)
    write_json(intensifier_path, intensifier)



# if __name__ == "__main__":
#     os.chdir(Path(__file__).resolve().parent.parent)
    
#     sanitize(Path("./hpc/logs/smac3_output/google_t5-efficient-small"), seed=123, min_budget=960_000, eta=4.805622828269508562053688, max_budget=512_000_000)

    