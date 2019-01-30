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

numerical = ['purchase_amount', 'numerical_1', 'numerical_1', 'installments']

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
    start = g.purchase_date.min()
    end = g.purchase_date.max()
    period = g.gap.mean()
    aggregated = {'card_id': g.iloc[0].card_id, 'start': start, 'end': end,
                  'duration': (end - start) / pd.Timedelta(hours=1) / 168,
                  'frequency': 1 / period if period > 0 else 1}

    # numerical features
    for column in numerical + lag:
        aggregated.update({'sum_' + column: g[column].sum(), 'min_' + column: g[column].min(),
                           'max_' + column: g[column].max(), 'mean_' + column: g[column].mean(),
                           'var_' + column: g[column].var()})

    # dummy encoding
    for column in categorical:
        aggregated.update(modify_keys(g[column].value_counts().to_dict(), column))

    # count all transactions
    aggregated.update({'transactions': len(g)})

    # get most frequent value
    aggregated.update(modify_keys(g[features].agg(most_frequent).to_dict(), 'most_frequent'))

    # count unique values
    aggregated.update(modify_keys(g[features].agg(nunique).to_dict(), 'nunique'))

    return aggregated


def main():
    """ Aggregate transactions by cards.
    """
    logger = logging.getLogger(__name__)
    logger.info('aggregate transactions chunks')

    merchants = pd.read_csv('./data/raw/merchants.csv')

    for chunk in range(8):
        gc.collect()
        logger.info(f'chunk {chunk!r}')
        transactions = read_chunk(chunk)
        logger.info(f'shape {transactions.shape!r}')

        logger.info(f'dates {chunk!r}')
        transactions.purchase_date = pd.to_datetime(transactions.purchase_date)
        transactions['day'] = transactions.purchase_date.apply(lambda x: x.day)
        transactions['week'] = transactions.purchase_date.apply(lambda x: ((x - datetime(x.year, 1, 1)).days // 7) + 1)
        transactions['month'] = transactions.purchase_date.apply(lambda x: x.month)
        transactions['year'] = transactions.purchase_date.apply(lambda x: x.year - 2016)

        logger.info(f'merge {chunk!r}')
        transactions_merchants = pd.merge(transactions, merchants, how='left',
                                          on=['merchant_id', 'subsector_id', 'merchant_category_id', 'city_id',
                                              'state_id'],
                                          suffixes=('_transaction', '_merchant'))

        logger.info(f'sort {chunk!r}')
        transactions_merchants.sort_values('purchase_date', inplace=True)

        aggregated_dicts = apply_parallel_without_concat(transactions_merchants.groupby(by='card_id'),
                                                         aggregate_transactions)

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
