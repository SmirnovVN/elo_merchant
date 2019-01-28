# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from random import shuffle


def read_chunk(chunk):
    df = pd.read_pickle('./data/interim/all_transactions_' + str(chunk) + '.pkl')
    return df


def main():
    """ Makes transactions chunks from historical_transactions and new_merchant_transactions and save it to interim.
    """
    logger = logging.getLogger(__name__)
    logger.info('making transactions chunks')

    transactions = pd.read_csv('./data/raw/historical_transactions.csv')
    transactions = pd.concat([transactions, pd.read_csv('./data/raw/new_merchant_transactions.csv')])
    cards = list(transactions.card_id.unique())
    shuffle(cards)
    size = (len(cards) + 8) // 8
    card_chunks = [set(cards[x:x + size]) for x in range(0, len(cards), size)]
    for i, chunk in enumerate(card_chunks):
        logger.info(f'chunk {i!r}: {len(chunk)!r}')
        transactions[transactions.card_id.isin(chunk)].to_pickle('./data/interim/all_transactions_' + str(i) + '.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
