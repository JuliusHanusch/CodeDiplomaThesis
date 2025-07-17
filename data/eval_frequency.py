import pandas as pd
from dateutil.relativedelta import *
from datetime import timedelta
import numpy as np


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

def make_equidistant_with_nans(df: pd.DataFrame, time_column: str, freq: str) -> pd.DataFrame:
    df_ = df.copy()
    df_[time_column] = pd.to_datetime(df_[time_column])
    df_ = df_.sort_values(time_column)
    
    full_index = pd.date_range(start=df_[time_column].min(),
                               end=df_[time_column].max(),
                               freq=freq)
    
    return df_.set_index(time_column).reindex(full_index).rename_axis(time_column).reset_index()

# returns true if temporal points can be considered to be equidistant (adjust thresholds as needed)
def eval_frequency(df: pd.DataFrame, time_column='datetime', min_df_len=64, debug=False) -> bool:
    amount_threshold = 0.8 # minimal share of precisely equidistant time data points
    total_time_ratio_threshold = 0.8 # ratio of most common time difference to average time difference
    amount_threshold_fixable = 0.60 # If below 60% Restorable -> Try luck with equidisation instead
    max_sample_count = 50_000

    date_time = df[time_column]
    if (len(date_time) < 2):
        return True, []
    date_time: pd.Series = sorted(date_time)
    date_time = np.array(date_time, dtype='datetime64[ms]')


    # ensure a sufficient share of time points is equally spaced
    ## count the number of occurrences of each time difference
    time_diff_dict: dict = {}
    sample_count = min(len(date_time)-1, max_sample_count) # if it is not clear after 10k we wont know it after 50k, 100k, ...
    for x in range(sample_count): 
        dt1 = date_time[x].astype('M8[ms]').astype('O')  # object = datetime.datetime
        dt2 = date_time[x+1].astype('M8[ms]').astype('O')
        relative_delta = relativedelta(dt2, dt1)
        time_diff_dict[relative_delta] = time_diff_dict.get(relative_delta, 0) + 1

    time_diff_dict = dict(sorted(time_diff_dict.items(), key=lambda item: item[1], reverse=True))
    most_common_time_diff = list(time_diff_dict.keys())[0]

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
    amount_share = time_diff_dict[most_common_time_diff] / max(sample_count, 1)


    # ensure we don't have too many gaps in the data
    ## calculate ratio of (most common time difference) / (average time delta per data point)
    first_time = date_time[0]
    last_time = date_time[-1]

    if first_time == last_time:
        return False, []
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
        
	
    # find fixable time series:
    # share of most frequent time distance is high enough,
    # but total time doesnt match (and ts is shorter as max_sample_count to avoid cutting already cutted TS)
    if (amount_share > amount_threshold_fixable
            and (time_diff_ratio < total_time_ratio_threshold or 1/max(time_diff_ratio,1E-5) < total_time_ratio_threshold))\
            and (time_diff_ratio != 0.0) and len(date_time-1) < max_sample_count:
        # attempting fix
        print("Fixable?")
        split_series = []
        tempdf_index = 0
        fixed_amount = 0
        for x in range(len(df) - 1):
            relative_delta = relativedelta(df[time_column][x + 1], df[time_column][x])
            if not (relative_delta == most_common_time_diff
                    or relative_delta == most_common_time_diff * 2 # single gaps are OK
                    or relative_delta == most_common_time_diff * 3 # double gaps are too
                    or relative_delta in to_del): #in to_del means it got clustered
                # split detected
                fixed_df = df.iloc[tempdf_index:x+1].reset_index(drop=True)
                if len(fixed_df) > min_df_len:
                    fixed_df_equi_max = make_equidistant_with_nans(fixed_df, time_column=time_column, freq=most_common_time_diff_time_delta)
                    split_series.append(fixed_df_equi_max)
                    fixed_amount += len(fixed_df)
                tempdf_index = x+1
        # TODO DRY
        fixed_df = df.iloc[tempdf_index:len(df)].reset_index(drop=True)
        if len(fixed_df) > min_df_len:
            fixed_df_equi_max = make_equidistant_with_nans(fixed_df, time_column=time_column, freq=most_common_time_diff_time_delta)
            split_series.append(fixed_df_equi_max)
            fixed_amount += len(fixed_df)
        # fixed_series = pd.concat(split_series).sort_values(by=time_column, inplace=False)

        # Check if we were truly able to fixe the desired amount (after applying min length)
        if fixed_amount / len(df) >= amount_threshold_fixable:
            return False, split_series
        else: # Try to equidise instead
            return False, []


    # true if both are above threshold
    return (amount_share >= amount_threshold
            and time_diff_ratio >= total_time_ratio_threshold
            and 1/max(time_diff_ratio,1E-5) >= total_time_ratio_threshold), []

