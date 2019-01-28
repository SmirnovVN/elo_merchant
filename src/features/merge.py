# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.features.aggregate import read_aggregated


def read_merged():
    df = pd.read_pickle('./data/interim/merged.pkl')
    return df


def main():
    """ Makes transactions chunks from historical_transactions and new_merchant_transactions and save it to interim.
    """
    logger = logging.getLogger(__name__)
    logger.info('aggregate transactions chunks')

    transactions = read_aggregated(0)
    for chunk in range(1, 8):
        logger.info(f'chunk {chunk!r}')
        transactions = pd.concat([transactions, read_aggregated(chunk)])

    merchants = pd.read_csv('./data/raw/merchants.csv')

    train = pd.read_csv('./data/raw/train.csv')

    test = pd.read_csv('./data/raw/test.csv')

    transactions_merchants = pd.merge(transactions, merchants, how='left',
                                      left_on=['most_frequent_merchant_id'], right_on=['merchant_id'])

    train = pd.merge(train, transactions_merchants, how='left', on=['card_id'])

    train['type'] = pd.Series('train')

    test = pd.merge(test, transactions_merchants, how='left', on=['card_id'])

    test['type'] = pd.Series('test')

    pd.concat([train, test]).to_pickle('./data/interim/merged.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
