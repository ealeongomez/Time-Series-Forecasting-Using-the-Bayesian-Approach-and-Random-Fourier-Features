import numpy as np
import pandas as pd
# import FIF
import warnings

warnings.filterwarnings('ignore')


def impute_missing_values(series, k=3):
    """Impute missing values in a series using neighboring average.

    Args:
        series (pd.Series): The series with missing values.
        k (int): The number of neighboring values to use for imputation.

    Returns:
        pd.Series: The series with imputed values.
    """

    # Find the indices of the missing values.
    missing_indices = series[series.isna()].index

    # For each missing value, impute it with the average of its k nearest neighbors.
    for index in missing_indices:
        neighbors = None
        if k >= index:
            neighbors = series[0:index + k].dropna()
        else:
            neighbors = series[index-k:index + k].dropna()
        series.loc[index] = neighbors.mean()

    return series


def create_features(sequence, n_steps):
    """ Create a dataframe from a series with n_steps lags """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
		# check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def split_data(series, tr_len):
    # series = np.array(series.tolist())
    val_len = (len(series) - tr_len) // 2
    tr_series = series[:tr_len]
    
    val_series = series[tr_len:(tr_len+val_len)]
    test_series = series[(tr_len+val_len):]
    
    return tr_series, val_series, test_series
