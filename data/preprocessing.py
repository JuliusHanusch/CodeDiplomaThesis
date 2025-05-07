import pandas as pd
import datasets as ds
import datetime as dt

from eval_frequency import eval_frequency


def preprocessing():
    dataset = ds.load_from_disk("data_sets_raw/Time_Corpus")
    dataset = dataset['train']

    count_total = len(dataset)
    count_equi = 0
    count_nonequi = 0

    list_dataset = []
    for set_data in dataset:
        df_data = pd.DataFrame()

        df_data['datetime'] = eval(set_data['date'])
        for i, values in enumerate(set_data['value']):
            df_data[f'value_{i}'] = values

        df_data = df_data[
            ~df_data.astype(str).apply(lambda row: row.str.contains('0000-01-01 00:00:00', case=False, na=False)).any(axis=1)].copy()
        try:
            df_data['datetime'] = df_data['datetime'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            df_data['datetime'] = df_data['datetime'].dt.round('us')
        except:
            print("skipping malformed row")
            continue
        equi_distance = eval_frequency(df_data)
        if equi_distance:
            count_equi += 1
            if len(df_data.columns) > 2:
                for column in df_data.columns[1:]:
                    df_set = pd.DataFrame()
                    df_set['datetime'] = df_data["datetime"]
                    df_set['value'] = df_data[column]

                    column_dataset = ds.Dataset.from_pandas(df_set)
                    list_dataset.append(column_dataset)
            else:
                list_dataset.append(ds.Dataset.from_pandas(df_data))
        else:
            count_nonequi += 1

    concatinated_dataset = ds.concatenate_datasets(list_dataset)
    dataset = ds.DatasetDict()
    dataset['train'] = concatinated_dataset

    dataset.save_to_disk("data_sets_processed/Time_Corpus")

    print(count_total)
    print(count_equi)
    print(count_nonequi)
    print(count_equi / count_total)


if __name__ == "__main__":
    preprocessing()
