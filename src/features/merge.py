# -*- coding: utf-8 -*-
import logging
import warnings
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.features.aggregate import read_aggregated

warnings.simplefilter(action='ignore', category=FutureWarning)


def read_merged():
    df = pd.read_pickle('./data/interim/merged.pkl')
    return df


def main():
    """ Merge all data.
    """
    logger = logging.getLogger(__name__)
    logger.info('merge transactions chunks')

    transactions = read_aggregated(0)
    for chunk in range(1, 8):
        logger.info(f'chunk {chunk!r}')
        transactions = pd.concat([transactions, read_aggregated(chunk)])

    train = pd.read_csv('./data/raw/train.csv')

    test = pd.read_csv('./data/raw/test.csv')

    train = pd.merge(train, transactions, how='left', on=['card_id'])

    train['type'] = pd.Series('train', index=train.index)
    train['outliers'] = 0
    train.loc[train['target'] < -30, 'outliers'] = 1

    test = pd.merge(test, transactions, how='left', on=['card_id'])

    test['type'] = pd.Series('test', index=test.index)

    assert train.shape[0] == 201917
    assert test.shape[0] == 123623

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
