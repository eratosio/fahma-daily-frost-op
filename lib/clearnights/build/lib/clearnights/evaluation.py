import pandas as pd
import numpy as np


def evaluation_tair_threshold(lst: pd.Series, bom_reference_tair: pd.DataFrame, threshold: float):
    """
    This evaluation is against air temperature. However, since this was developed a method to compare against actual cloud observations have been developed using the BoM ceilometer data. But this method has not yet been implemented in the evaluation module.
    :param lst:
    :param bom_reference_tair:
    :param threshold:
    :return:
    """

    lst = lst.rename('lst')

    # add tair data to output dataframe. Use nearest neighbour to alignment time index
    merged_df = pd.merge_asof(lst, bom_reference_tair, left_index=True, right_index=True, direction='nearest')

    daily_min = merged_df.groupby(merged_df.index.date).agg(
        {'air_temp': [np.nanmin], 'lst': [np.nanmin]})
    daily_min.columns = ['_'.join(c) for c in daily_min.columns]

    frost_days_by_tair = daily_min.loc[daily_min.air_temp_nanmin <= threshold]
    non_frost_days_tair = daily_min.loc[daily_min.air_temp_nanmin > threshold]

    frost_days_by_stage1_lst = daily_min.loc[daily_min.lst_nanmin <= threshold]
    non_frost_days_by_stage1_lst = daily_min.loc[daily_min.lst_nanmin > threshold]

    # how many LST frost days which are not frost days by air
    false_positives = len(set(frost_days_by_stage1_lst.index).difference(frost_days_by_tair.index))

    # how many tair frost days which are not frost days by LST
    false_negatives = len(set(frost_days_by_tair.index).difference(frost_days_by_stage1_lst.index))

    true_positives = len(set(frost_days_by_stage1_lst.index).intersection(frost_days_by_tair.index))
    true_negatives = len(set(non_frost_days_tair.index).intersection(non_frost_days_by_stage1_lst.index))
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = None
    precision = true_positives / (true_positives + false_positives)
    total_positives = true_positives + false_positives
    total_negatives = true_negatives + false_negatives
    total = total_negatives + total_positives
    overall_accuracy = (true_positives + true_negatives) / total

    results = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_positives,
        'false_negatives': false_negatives,
        'total_positive': total_positives,
        'total_negatives': total_negatives,
        'overall_accuracy': overall_accuracy,
        'recall': recall,
        'precision': precision
    }

    results_df = pd.DataFrame.from_dict([results], orient='columns')

    # Skikitlearn report
    # frost_by_tair = daily_min.air_temp_nanmin <= threshold
    # frost_by_lst = daily_min.lst_nanmin <= threshold
    # scikitlearn_results = pd.DataFrame.from_dict(classification_report(frost_by_tair, frost_by_lst, target_names=['non-frost', 'frost'], output_dict=True))

    return results_df
