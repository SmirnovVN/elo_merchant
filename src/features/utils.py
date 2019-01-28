import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm


def label_encoding(df, features):
    """Get mapped by label encoding columns and encoders"""
    mapped = {}
    encoders = {}
    for f in features:
        encoders[f] = LabelEncoder()
        encoders[f].fit(np.unique(df[f].astype(str).values))
        mapped[f] = encoders[f].transform(df[f].astype(str).values)
    return mapped, encoders


def merge_unique(series):
    """Replace rare values to 'unique' """
    counts = series.value_counts()
    unique = set(counts[counts < 30].index)
    return series.apply(lambda x: 'unique' if x in unique else x)


def most_frequent(series):
    """Get most frequent value"""
    counts = series.value_counts()
    if not counts.empty:
        return counts.index[0]


def nunique(series):
    """Count unique values"""
    counts = series.value_counts()
    if not counts.empty:
        return counts.count()


def get_sequence(df, column, n):
    """Extract sequence of values to n features"""
    series = df[column]
    result = {}
    value = None
    if not series.empty:
        last = -1
        j = 0
        for i, value in series.iteritems():
            if value != last:
                last = value
                j += 1
                if j < n:
                    result[column + str(j)] = value
        result[column + '_last'] = value
    return result


def modify_keys(dictionary, modifier):
    """Append modifier to each key from dict"""
    result = {}
    for key, value in dictionary.items():
        result[modifier + str(key)] = value
    return result


def apply_parallel(grouped, func):
    """Apply aggregation function on each group and concatenate to one DataFrame with multiprocessing"""
    return pd.concat(
        Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in tqdm(grouped)))


def apply_parallel_sequential(grouped, func):
    """Apply aggregation function on each group and concatenate to one DataFrame with multiprocessing"""
    result = pd.DataFrame()
    for item in tqdm(Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(func)(group) for name, group in tqdm(grouped))
    ):
        result = pd.concat([result, item])
    return result


def apply_parallel_without_concat(grouped, func, n_jobs=None):
    """Apply aggregation function on each group with multiprocessing"""
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    return Parallel(n_jobs=n_jobs)(delayed(func)(group) for name, group in tqdm(grouped))
