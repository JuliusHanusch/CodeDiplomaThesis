import pandas as pd

# returns true if the frequency of the data is equidistant
def eval_frequency(df: pd.DataFrame) -> bool:
    amount_threshold = 0.9 # minimal share of equidistant time data points
    total_time_ratio_threshold = 0.9 # ratio of most common time difference to average time difference

    datetime = df['datetime']

    # calculate number of occurrences of each time difference
    time_diff_dict: dict = {}
    for x in range(len(datetime)-1):
        diff = datetime[x+1] - datetime[x]
        time_diff_dict[diff] = time_diff_dict.get(diff, 0) + 1
    time_diff_dict = dict(sorted(time_diff_dict.items(), key=lambda item: item[1], reverse=True))
    most_common_time_diff = list(time_diff_dict.keys())[0]
    amount_share = time_diff_dict[most_common_time_diff] / len(datetime)

    # calculate ratio of (most common time difference) / (average time delta per data point)
    first_time = datetime.iloc[0]
    last_time = datetime.iloc[-1]
    average_time_diff_per_data_point = ((last_time - first_time) / time_diff_dict[most_common_time_diff])
    time_diff_ratio = most_common_time_diff / average_time_diff_per_data_point

    # true if both are above threshold
    return amount_share > amount_threshold and time_diff_ratio > total_time_ratio_threshold

