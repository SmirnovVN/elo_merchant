# -*- coding: utf-8 -*-
import logging
import gc
from datetime import datetime

import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data import read_chunk
from src.features.utils import apply_parallel_without_concat, modify_keys, most_frequent, nunique

features = ['city_id', 'installments', 'merchant_id', 'state_id', 'subsector_id',
            'month_lag', 'day', 'week', 'month', 'year']

numerical = ['purchase_amount', 'numerical_1', 'numerical_2', 'installments', 'day', 'week', 'month', 'year',
             'month_diff', 'month_diff_lagged', 'most_recent_sales_range_num', 'most_recent_purchases_range_num',
             'price', 'numerical_1_p_2', 'numerical_1_m_2', 'numerical_1_s_2', 'numerical_1_d_2']

lag = ['month_lag', 'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6', 'avg_sales_lag12',
       'avg_purchases_lag12', 'active_months_lag12']

categorical = ['authorized_flag', 'month_lag', 'category_1_transaction', 'category_1_merchant',
               'category_2_transaction', 'category_2_merchant', 'category_3', 'category_4',
               'most_recent_sales_range', 'most_recent_purchases_range']


def read_aggregated(chunk):
    df = pd.read_pickle('./data/interim/aggregated_' + str(chunk) + '.pkl')
    return df


def aggregate_transactions(g):
    g['shifted'] = g.purchase_date.shift(1)
    g['gap'] = (g.purchase_date - g.shifted) / pd.Timedelta(hours=1) / 168

    aggregated = dict()
    aggregated['card_id'] = g.iloc[0].card_id
    aggregated['start'] = g.purchase_date.min()
    aggregated['end'] = g.purchase_date.max()
    aggregated['period'] = g.gap.mean()
    aggregated['duration'] = (aggregated['end'] - aggregated['start']) / pd.Timedelta(hours=1) / 168
    aggregated['frequency'] = 1 / aggregated['period'] if aggregated['period'] > 0 else 1
    aggregated['transactions'] = len(g)

    # numerical features
    for column in numerical + lag:
        aggregated.update({'sum_' + column: g[column].sum(), 'min_' + column: g[column].min(),
                           'max_' + column: g[column].max(), 'mean_' + column: g[column].mean(),
                           'var_' + column: g[column].var()})

    # dummy encoding
    for column in categorical:
        aggregated.update(modify_keys(g[column].value_counts().to_dict(), column))

    # combo features
    aggregated['installments_by_day'] = aggregated['sum_installments'] / aggregated['duration']
    aggregated['transactions_by_day'] = aggregated['transactions'] / aggregated['duration']
    aggregated['purchase_amount_by_day'] = aggregated['sum_purchase_amount'] / aggregated['duration']
    aggregated['price_by_day'] = aggregated['sum_price'] / aggregated['duration']
    aggregated['purchase_amount_by_transactions'] = aggregated['sum_purchase_amount'] / aggregated['transactions']

    all_features = list(set(features + numerical + lag + categorical))

    # get most frequent value
    aggregated.update(modify_keys(g[all_features].agg(most_frequent).to_dict(), 'most_frequent'))

    # count unique values
    aggregated.update(modify_keys(g[all_features].agg(nunique).to_dict(), 'nunique'))

    return aggregated


def main():
    """ Aggregate transactions by cards.
    """
    logger = logging.getLogger(__name__)
    logger.info('aggregate transactions chunks')

    merchants = pd.read_csv('./data/raw/merchants.csv')
    merchants['most_recent_sales_range_num'] = merchants['most_recent_sales_range'].map(
        {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}).astype(int)
    merchants['most_recent_purchases_range_num'] = merchants['most_recent_purchases_range'].map(
        {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}).astype(int)

    for chunk in range(8):
        gc.collect()
        logger.info(f'chunk {chunk!r}')
        transactions = read_chunk(chunk)
        logger.info(f'shape {transactions.shape!r}')

        logger.info(f'fill nan {chunk!r}')
        transactions['category_2'].fillna(1.0, inplace=True)
        transactions['category_3'].fillna('A', inplace=True)
        transactions['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
        transactions['installments'].replace(-1, pd.np.nan, inplace=True)
        transactions['installments'].replace(999, pd.np.nan, inplace=True)

        logger.info(f'dates {chunk!r}')
        transactions.purchase_date = pd.to_datetime(transactions.purchase_date)
        transactions['day'] = transactions.purchase_date.apply(lambda x: x.day)
        transactions['week'] = transactions.purchase_date.apply(lambda x: ((x - datetime(x.year, 1, 1)).days // 7) + 1)
        transactions['month'] = transactions.purchase_date.apply(lambda x: x.month)
        transactions['year'] = transactions.purchase_date.apply(lambda x: x.year - 2016)
        transactions['month_diff'] = (datetime.today() - transactions['purchase_date']).dt.days / 30
        transactions['month_diff_lagged'] = transactions['month_diff'] + transactions['month_lag']
        transactions['amount_month_ratio'] = transactions['purchase_amount'] / transactions['month_diff']

        logger.info(f'merge {chunk!r}')
        transactions.reset_index(inplace=True)
        transactions['ident'] = transactions.index
        t_m = pd.merge(transactions, merchants, how='left',
                       on=['merchant_id', 'subsector_id', 'merchant_category_id', 'city_id',
                           'state_id'],
                       suffixes=('_transaction', '_merchant'))

        logger.info(f'shape {t_m.shape!r}')
        t_m.drop_duplicates(subset=['ident'], keep='last', inplace=True)
        logger.info(f'shape {t_m.shape!r}')
        logger.info(f'indexes {t_m.ident.nunique()!r}')

        logger.info(f'sort {chunk!r}')
        t_m.sort_values('purchase_date', inplace=True)

        logger.info(f'features {chunk!r}')
        t_m['price'] = t_m['purchase_amount'] / t_m['installments']
        t_m['numerical_1_p_2'] = t_m['numerical_1'] + t_m['numerical_2']
        t_m['numerical_1_s_2'] = t_m['numerical_1'] - t_m['numerical_2']
        t_m['numerical_1_m_2'] = t_m['numerical_1'] * t_m['numerical_2']
        t_m['numerical_1_d_2'] = t_m['numerical_1'] / t_m['numerical_2']

        aggregated_dicts = apply_parallel_without_concat(t_m.groupby(by='card_id'), aggregate_transactions)

        pd.DataFrame(aggregated_dicts).to_pickle('./data/interim/aggregated_' + str(chunk) + '.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
