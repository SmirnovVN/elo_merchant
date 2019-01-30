# -*- coding: utf-8 -*-
import logging

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

    start_min = df.start.min()
    end_min = df.end.min()
    df.first_active_month = pd.to_datetime(df.first_active_month)
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
