from constants import *
from scaling_laws import load_data

import ConfigSpace as CS
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)
from deepcave import Recorder, Objective
from deepcave.runs import Status
from deepcave.runs import Run

from src.search_space import get_config_space


if __name__ == "__main__":
    results = load_data()
    results_hp = results['config'].apply(json.loads).apply(pd.Series)
    results_hp = results_hp.drop(["seed"], axis="columns")
    features = list(results_hp.columns)
    results = pd.concat([results.drop(columns='config'), results_hp], axis=1)
    results = results.drop(["seed"], axis="columns")
    print(results)
    cs = get_config_space(
        training_data_paths=1
    )
    run = Run(name="my_trials", configspace=cs)

    # Define objectives
    loss = Objective(metric, optimize="lower")
    # loss = Objective("loss", lower=0, optimize="lower")
    time = Objective("time", lower=0, optimize="lower")

    # Define budgets
    max_epochs = 8
    n_epochs = 4
    budgets = np.linspace(0, max_epochs, num=n_epochs)

    # Others
    num_configs = 1000
    num_runs = 3
    save_path = "logs/DeepCAVE/mnist_pytorch"

    for run_id in range(num_runs):
        random.seed(run_id)
        configspace = get_configspace(run_id)

        with Recorder(configspace, objectives=[accuracy, loss, time], save_path=save_path) as r:
            r.