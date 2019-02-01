# -*- coding: utf-8 -*-
import logging
from datetime import datetime

import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.features.merge import read_merged


def read_test():
    df = pd.read_pickle('./data/processed/test.pkl')
    return df


def read_train():
    df = pd.read_pickle('./data/processed/train.pkl')
    return df


def main():
    """ Aggregate transactions by cards.
    """
    logger = logging.getLogger(__name__)
    logger.info('build features and split back')

    df = read_merged()

    categorical = ['most_frequent_authorized_flag', 'most_frequent_category_1_merchant',
                   'most_frequent_category_1_transaction', 'most_frequent_category_3',
                   'most_frequent_category_4', 'most_frequent_most_recent_purchases_range',
                   'most_frequent_most_recent_sales_range']

    df = pd.get_dummies(df, columns=categorical)

    df.first_active_month = pd.to_datetime(df.first_active_month)

    # datetime features
    df['quarter'] = df['first_active_month'].dt.quarter
    df['elapsed_time'] = (datetime.today() - df['first_active_month']).dt.days

    df['days_feature1'] = df['elapsed_time'] * df['feature_1']
    df['days_feature2'] = df['elapsed_time'] * df['feature_2']
    df['days_feature3'] = df['elapsed_time'] * df['feature_3']

    df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
    df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
    df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']

    for f in ['feature_1', 'feature_2', 'feature_3']:
        order_label = df.groupby([f])['outliers'].mean()
        df[f] = df[f].map(order_label)

    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
    df['feature_mean'] = df['feature_sum'] / 3
    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

    start_min = df.start.min()
    end_min = df.end.min()
    first_min = df.first_active_month.min()

    df.start = df.start.apply(lambda x: (x - start_min) / pd.Timedelta(days=1))
    df.end = df.end.apply(lambda x: (x - end_min) / pd.Timedelta(days=1))
    df.first_active_month = df.first_active_month.apply(lambda x: (x - first_min) / pd.Timedelta(days=30))

    df.set_index('card_id', inplace=True)

    df[df.type == 'train'].drop(columns=['most_frequent_merchant_id', 'type']).to_pickle('./data/processed/train.pkl')
    df[df.type == 'test'].drop(columns=['most_frequent_merchant_id', 'type']).to_pickle('./data/processed/test.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
