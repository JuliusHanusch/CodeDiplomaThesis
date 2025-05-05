import pandas as pd
from dateutil.relativedelta import *

# catch edge case where data points are distanced monthly (months have different numbers of days)
def month_edgecase(date_time, threshold, debug) -> bool:
    month_differences = {}
    for x in range(len(date_time)-1):

        i = 0
        earlier = date_time[x]
        later = date_time[x+1]

        while earlier < later:
            i += 1
            earlier = earlier + relativedelta(months=1)
            # if time delta is greater than 1 month, stop edge case test and resume normally (return False)
            if i == 1 and earlier > later:
                if debug:
                    print("Distance < 1 month, quitting month_edgecase test")
                return False

        if earlier > later:
            return False

        month_differences[i] = month_differences.get(i, 0) + 1
        month_differences = dict(sorted(month_differences.items(), key=lambda item: item[1], reverse=True))

    if debug:
        print("month_diff dict", month_differences)

    return (month_differences[list(month_differences.keys())[0]] / len(date_time)) > threshold


# returns true if temporal points can be considered to be equidistant (adjust thresholds as needed)
def eval_frequency(df: pd.DataFrame, debug=False) -> bool:
    amount_threshold = 0.9 # minimal share of precisely equidistant time data points
    total_time_ratio_threshold = 0.9 # ratio of most common time difference to average time difference

    date_time = df['datetime']

    date_time = sorted(date_time)
    print(date_time)

    # count the number of occurrences of each time difference
    time_diff_dict: dict = {}
    for x in range(len(date_time)-1):
        diff = date_time[x+1] - date_time[x]
        time_diff_dict[diff] = time_diff_dict.get(diff, 0) + 1
    time_diff_dict = dict(sorted(time_diff_dict.items(), key=lambda item: item[1], reverse=True))
    most_common_time_diff = list(time_diff_dict.keys())[0]
    amount_share = time_diff_dict[most_common_time_diff] / len(date_time)

    if month_edgecase(date_time, amount_threshold, debug):
        return True

    # calculate ratio of (most common time difference) / (average time delta per data point)
    first_time = date_time[0]
    last_time = date_time[-1]
    average_time_diff_per_data_point = ((last_time - first_time) / time_diff_dict[most_common_time_diff])
    time_diff_ratio = most_common_time_diff / average_time_diff_per_data_point

    if debug:
        print("Amount share",amount_share, ("above" if (amount_share>amount_threshold) else "below"), "threshold")
        print(time_diff_dict)
        print("Timediff ratio", time_diff_ratio, ("above" if (time_diff_ratio > total_time_ratio_threshold) else "below"), "threshold")

    # true if both are above threshold
    return amount_share > amount_threshold and time_diff_ratio > total_time_ratio_threshold

