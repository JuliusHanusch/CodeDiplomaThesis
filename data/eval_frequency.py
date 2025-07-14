import pandas as pd
from dateutil.relativedelta import *
from datetime import timedelta
import numpy as np


def relative_delta_to_time_delta(relative_delta: relativedelta) -> timedelta:
    return timedelta(
        # months are approximated as an average of 30d10h
        days=relative_delta.days + relative_delta.months * 30 + relative_delta.weeks * 7 + relative_delta.years * 365,
        hours=relative_delta.hours + relative_delta.months * 10,
        minutes=relative_delta.minutes,
        seconds=relative_delta.seconds,
        milliseconds=int(relative_delta.microseconds / 1000),
        microseconds=relative_delta.microseconds % 1000,
    )

# returns true if temporal points can be considered to be equidistant (adjust thresholds as needed)
def eval_frequency(df: pd.DataFrame, debug=False) -> bool:
    amount_threshold = 0.9 # minimal share of precisely equidistant time data points
    total_time_ratio_threshold = 0.9 # ratio of most common time difference to average time difference

    date_time = df['datetime']
    if (len(date_time) < 2):
        return True
    date_time: pd.Series = sorted(date_time)
    date_time = np.array(date_time, dtype='datetime64[ms]')


    # ensure a sufficient share of time points is equally spaced
    ## count the number of occurrences of each time difference
    time_diff_dict: dict = {}
    sample_count = min(len(date_time)-1, 10_000) # if it is not clear after 10k we wont know it after 50k, 100k, ...
    for x in range(sample_count): 
        dt1 = date_time[x].astype('M8[ms]').astype('O')  # object = datetime.datetime
        dt2 = date_time[x+1].astype('M8[ms]').astype('O')
        relative_delta = relativedelta(dt2, dt1)
        time_diff_dict[relative_delta] = time_diff_dict.get(relative_delta, 0) + 1

    time_diff_dict = dict(sorted(time_diff_dict.items(), key=lambda item: item[1], reverse=True))
    most_common_time_diff = list(time_diff_dict.keys())[0]
    amount_share = time_diff_dict[most_common_time_diff] / max(sample_count, 1)


    # ensure we don't have too many gaps in the data
    ## calculate ratio of (most common time difference) / (average time delta per data point)
    first_time = date_time[0]
    last_time = date_time[-1]

    if first_time == last_time:
        return False
        #raise Exception("Dataset malformed")

    if isinstance(first_time, pd.Timestamp):
        average_time_diff_per_data_point = (last_time.to_pydatetime() - first_time.to_pydatetime()) / max(sample_count, 1)
    else:
        average_time_diff_per_data_point = (last_time - first_time) / max(sample_count, 1)
    
    average_time_diff_per_data_point = max(
        np.timedelta64(1, 'ms'),
        average_time_diff_per_data_point
    )

    time_diff_ratio = relative_delta_to_time_delta(most_common_time_diff) / average_time_diff_per_data_point.astype('O')

    if debug:
        print(">>> eval_frequency DEBUG OUTPUT")
        print("date_time", date_time)
        print("time_diff_dict", time_diff_dict)
        print("Amount share",amount_share, ("above" if (amount_share>amount_threshold) else "below"), "threshold")
        print("Timediff ratio", time_diff_ratio, ("above" if (time_diff_ratio > total_time_ratio_threshold) else "below"), "threshold")
        print("Reciprocal Timediff ratio", 1/max(time_diff_ratio,1E-5), ("above" if ((1/max(time_diff_ratio,1E-5)) > total_time_ratio_threshold) else "below"),
          "threshold")

    # true if both are above threshold
    return (amount_share > amount_threshold
            and time_diff_ratio > total_time_ratio_threshold
            and 1/max(time_diff_ratio,1E-5) > total_time_ratio_threshold)

