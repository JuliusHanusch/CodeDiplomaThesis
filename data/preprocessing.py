import pandas as pd
import datasets as ds
import datetime as dt

from eval_frequency import eval_frequency


def preprocessing():
    dataset = ds.load_from_disk("data_sets_raw/Time_Corpus")
    dataset = dataset["train"]

    count_total = len(dataset)
    count_equi = 0
    count_nonequi = 0

    list_dataset = []
    case_study = {"EmptyValueColumn": [], "MalformedTimeColumn": []}
    for set_data in dataset:
        if not set_data["value"]:
            case_study["EmptyValueColumn"].append(set_data["name"])
            continue
        name = set_data["name"]

        dict_dataset = {"datetime": eval(set_data["date"])}
        for i, values in enumerate(set_data["value"]):
            dict_dataset[f"value_{i}"] = values

        df_data = pd.DataFrame.from_dict(dict_dataset)

        df_data = df_data[
            ~df_data.astype(str)
            .apply(
                lambda row: row.str.contains(
                    "0000-01-01 00:00:00", case=False, na=False
                )
            )
            .any(axis=1)
        ].copy()
        try:
            df_data["datetime"] = df_data["datetime"].apply(
                lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            )
            df_data["datetime"] = df_data["datetime"].dt.round("us")
        except:
            case_study["MalformedTimeColumn"].append(name)
            print("skipping malformed row")
            continue
        equi_distance = eval_frequency(df_data)
        if equi_distance:
            count_equi += 1
            if len(df_data.columns) > 2:
                for column in df_data.columns[1:]:
                    df_set = pd.DataFrame()
                    df_set["datetime"] = df_data["datetime"]
                    df_set["value"] = df_data[column]

                    dict_set = {
                        "identifier": [f"{name}_{column}"],
                        "datetime": [list(df_set["datetime"])],
                        "value": [list(df_set["value"])],
                    }
                    column_dataset = ds.Dataset.from_dict(dict_set)
                    list_dataset.append(column_dataset)
            else:
                df_data.rename(columns={"value_0": "value"}, inplace=True)
                dict_data = {
                    "identifier": [name],
                    "datetime": [list(df_data["datetime"])],
                    "value": [list(df_data["value"])],
                }
                univar_dataset = ds.Dataset.from_dict(dict_data)
                list_dataset.append(univar_dataset)
        else:
            count_nonequi += 1

    concatinated_dataset = ds.concatenate_datasets(list_dataset)
    dataset = ds.DatasetDict()
    dataset["train"] = concatinated_dataset

    dataset.save_to_disk("data_sets_raw/Time_Corpus_Processed")

    print(count_equi)
    print(count_nonequi)
    print(count_equi / count_total)
    print(case_study)


if __name__ == "__main__":
    preprocessing()
