# -*- coding: utf-8 -*-
import logging
from datetime import datetime

import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data import read_chunk
from src.features.utils import apply_parallel_without_concat, modify_keys, most_frequent, nunique, get_sequence

features = ['city_id', 'installments', 'merchant_id', 'state_id', 'subsector_id',
            'month_lag', 'day', 'week', 'month', 'year']


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
                  'frequency': 1 / period if period > 0 else 1, 'sum_amount': g.purchase_amount.sum(),
                  'min_amount': g.purchase_amount.min(), 'max_amount': g.purchase_amount.max(),
                  'mean_amount': g.purchase_amount.mean()}

    # count all transactions
    aggregated.update({'transactions': len(g)})

    # get most frequent value
    aggregated.update(
        modify_keys(g[features].agg(most_frequent).to_dict(), 'most_frequent_'))

    # count unique values
    aggregated.update(modify_keys(g[features].agg(nunique).to_dict(), 'nunique_'))

    # count events like a click, hover etc.
    aggregated.update(modify_keys(g.authorized_flag.value_counts().to_dict(), 'authorized_flag_'))
    aggregated.update(modify_keys(g.month_lag.value_counts().to_dict(), 'month_lag_'))
    aggregated.update(modify_keys(g.category_1.value_counts().to_dict(), 'category_1_'))
    aggregated.update(modify_keys(g.category_2.value_counts().to_dict(), 'category_2_'))
    aggregated.update(modify_keys(g.category_3.value_counts().to_dict(), 'category_3_'))

    return aggregated


def main():
    """ Aggregate transactions by cards.
    """
    logger = logging.getLogger(__name__)
    logger.info('aggregate transactions chunks')

    for chunk in range(8):
        logger.info(f'chunk {chunk!r}')
        transactions = read_chunk(chunk)

        transactions.purchase_date = pd.to_datetime(transactions.purchase_date)
        transactions['day'] = transactions.purchase_date.apply(lambda x: x.day)
        transactions['week'] = transactions.purchase_date.apply(lambda x: ((x - datetime(x.year, 1, 1)).days // 7) + 1)
        transactions['month'] = transactions.purchase_date.apply(lambda x: x.month)
        transactions['year'] = transactions.purchase_date.apply(lambda x: x.year - 2016)

        transactions.sort_values('purchase_date', inplace=True)

        aggregated_dicts = apply_parallel_without_concat(transactions.groupby(by='card_id'), aggregate_transactions)

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
