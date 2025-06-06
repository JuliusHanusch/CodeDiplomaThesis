import pandas as pd
from dateutil.relativedelta import *
from datetime import timedelta

# needed for further calculations with time
# relative delta will have non-exact values like "month"
# time delta is an absolute value
def relative_delta_to_time_delta(relative_delta: relativedelta) -> timedelta:
    return timedelta(
        # months are approximated as an average of 30d10h
        days=relative_delta.days + relative_delta.months * 30 + relative_delta.years * 365,
        hours=relative_delta.hours + relative_delta.months * 10,
        minutes=relative_delta.minutes,
        seconds=relative_delta.seconds,
        milliseconds=int(relative_delta.microseconds / 1000),
        microseconds=relative_delta.microseconds % 1000,
    )

# returns true if temporal points can be considered to be equidistant (adjust thresholds as needed)
def eval_frequency(df: pd.DataFrame, debug=False) -> (bool, list):
    if debug: print(">>> eval_frequency DEBUG OUTPUT")

    amount_threshold = 0.9 # minimal share of precisely equidistant time data points
    total_time_ratio_threshold = 0.9 # ratio of most common time difference to average time difference
    amount_threshold_fixable = 0.8
    min_df_len = 64

    date_time = df['datetime']
    if len(date_time) < 2:
        if debug: print("Dataset too short")
        return True, []
    date_time = sorted(date_time)

    # ensure a sufficient share of time points is equally spaced
    ## count the number of occurrences of each time difference
    time_diff_dict: dict = {}
    for x in range(len(date_time)-1):
        relative_delta = relativedelta(date_time[x+1], date_time[x])
        time_diff_dict[relative_delta] = time_diff_dict.get(relative_delta, 0) + 1

    time_diff_dict = dict(sorted(time_diff_dict.items(), key=lambda item: item[1], reverse=True))
    most_common_time_diff = list(time_diff_dict.keys())[0]

    if debug: print("Pre-Clustering", time_diff_dict)
    # If most common timediff == 0
    if most_common_time_diff == relativedelta():
        if debug: print("Dataset malformed: most common time diff 0")
        return False, []

    # cluster time deltas that are close to most common time diff
    to_del = []
    most_common_time_diff_time_delta = relative_delta_to_time_delta(most_common_time_diff)
    for ditem in list(time_diff_dict.keys())[1:]:
        if ditem != relativedelta():
            if ((relative_delta_to_time_delta(ditem) / most_common_time_diff_time_delta > amount_threshold)
                and (most_common_time_diff_time_delta / relative_delta_to_time_delta(ditem) > amount_threshold)):
                if debug: print("clustering", most_common_time_diff_time_delta, "item:", ditem, relative_delta_to_time_delta(ditem))
                time_diff_dict[most_common_time_diff] = time_diff_dict[most_common_time_diff] + time_diff_dict[ditem]
                time_diff_dict[ditem] = 0
                to_del.append(ditem)
            else:
                if debug: print("NOT clustering", most_common_time_diff_time_delta, "item:", ditem, relative_delta_to_time_delta(ditem))
    for del_item in to_del:
        del time_diff_dict[del_item]
    amount_share = time_diff_dict[most_common_time_diff] / max(len(date_time) - 1, 1)

    # ensure we don't have too many gaps in the data
    ## calculate ratio of (most common time difference) / (average time delta per data point)
    first_time = date_time[0]
    last_time = date_time[-1]

    if first_time == last_time:
        if debug: print("Dataset malformed: first and last time are equal")
        return False, []
        #raise Exception("Dataset malformed")

    if isinstance(first_time, pd.Timestamp):
        average_time_diff_per_data_point = (last_time.to_pydatetime() - first_time.to_pydatetime()) / max(len(date_time) - 1, 1)
    else:
        average_time_diff_per_data_point = (last_time - first_time) / max(len(date_time) - 1, 1)

    time_diff_ratio = most_common_time_diff_time_delta / average_time_diff_per_data_point

    if debug:
        #print("date_time", date_time)
        print("time_diff_dict", time_diff_dict)
        print("Amount share",amount_share, ("above" if (amount_share>amount_threshold) else "below"), "threshold")
        print("Timediff ratio", time_diff_ratio, ("above" if (time_diff_ratio > total_time_ratio_threshold) else "below"), "threshold")
        print("Reciprocal Timediff ratio", 1/max(time_diff_ratio,1E-5), ("above" if ((1/max(time_diff_ratio,1E-5)) > total_time_ratio_threshold) else "below"),
          "threshold")

    # find fixable time series:
    # share of most frequent time distance is high enough,
    # but total time doesnt match
    if (amount_share > amount_threshold_fixable
            and (time_diff_ratio < total_time_ratio_threshold or 1/max(time_diff_ratio,1E-5) < total_time_ratio_threshold))\
            and (time_diff_ratio != 0.0):
        # attempting fix

        print("Fixable?")

        split_series = []

        tempdf_index = 0
        for x in range(len(df) - 1):
            relative_delta = relativedelta(df['datetime'][x + 1], df['datetime'][x])
            if not (relative_delta == most_common_time_diff
                    or relative_delta == most_common_time_diff * 2 # single gaps are OK
                    or relative_delta in to_del): #in to_del means it got clustered
                # split detected
                if x+1 - tempdf_index > min_df_len:
                    split_series.append(df.iloc[tempdf_index:x+1].reset_index(drop=True))
                tempdf_index = x+1

        if len(df) - tempdf_index > min_df_len:
            split_series.append(df.iloc[tempdf_index:len(df)].reset_index(drop=True))

        return False, split_series



    # true if both are above threshold
    return (amount_share > amount_threshold
            and time_diff_ratio > total_time_ratio_threshold
            and 1/max(time_diff_ratio,1E-5) > total_time_ratio_threshold), []

