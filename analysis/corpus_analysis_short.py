import datasets as ds
import os
from tqdm.auto import tqdm


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    uci = ds.load_from_disk("../data/data_sets_raw/UCI_Corpus")
    kaggle = ds.load_from_disk("../data/data_sets_raw/Time_Corpus")

    for corpus, cname in [(kaggle, "kaggle"), (uci, "uci")]:
        print(cname)
        train = corpus["train"]
        test = corpus["test"] 
        datasets = set(train["name"] + test["name"])
        # print(datasets)
        print("Number of Datasets: ", len(datasets))

        timestamps = []
        datapoints = []
        timeseries = []
        for split in [train, test]:
            for dataset in tqdm(split):
                values = dataset["value"]
                number_ts = len(values)
                timeseries.append(number_ts)
                if number_ts == 0:
                    continue
                timestamps.append(len(values[0]))
                datapoints.append(sum((len(v) for v in values)))


        print("Number of TS: ", sum(timeseries))
        print("Number of Timestamps: ", sum(timestamps))
        print("Number of Datapoints: ", sum(datapoints))


