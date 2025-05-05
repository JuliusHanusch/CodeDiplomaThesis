import os
import pathlib

import pandas as pd
import datasets as ds

from typer import Typer

from eval_frequency import eval_frequency


def preprocessing():
    dataset = ds.load_from_disk("data_sets_raw/Time_Corpus")
    dataset = dataset['train']
    count_complete = len(dataset)

    dict_dataset = {}
    count_equi = 0
    for set_data in dataset:
        df_data = pd.DataFrame()

        df_data['datetime'] = eval(set_data['date'])
        for i, values in enumerate(set_data['value']):
            df_data[f'value_{i}'] = values

        df_data['datetime'] = pd.to_datetime(df_data['datetime'])

        equi_distance = eval_frequency(df_data)

        if equi_distance:
            count_equi += 1
            if len(df_data.columns) > 2:
                for column in df_data.columns[1:]:
                    df_set = pd.DataFrame()
                    df_set['datetime'] = df_data["datetime"]
                    df_set[column] = df_data[column]

                    column_dataset = ds.Dataset.from_pandas(df_set)
            else:
                break


if __name__ == "__main__":
    preprocessing()
