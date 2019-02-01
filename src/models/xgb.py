import gc
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
from sklearn.metrics import mean_squared_error
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pandas.core.common import SettingWithCopyWarning
from src.features import read_train, read_test

pd.options.display.max_columns = None
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

FEATS_EXCLUDED = ['target', 'outliers']


# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def kfold_xgboost(train, test, num_folds, stratified=False):
    target = train['target']
    logger = logging.getLogger(__name__)
    logger.info("Starting XGBoost. Train shape: {}, test shape: {}".format(train.shape, test.shape))

    xgb_params = {'eta': 0.001, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True}

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=8)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=8)

    oof_preds = np.zeros(len(train))
    sub_preds = np.zeros(len(test))

    feats = [f for f in train.columns if f not in FEATS_EXCLUDED]

    num_round = 2000

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
        trn_data = xgb.DMatrix(data=train[feats].iloc[trn_idx], label=target.iloc[trn_idx])
        val_data = xgb.DMatrix(data=train[feats].iloc[val_idx], label=target.iloc[val_idx])
        watchlist = [(trn_data, 'train'), (val_data, 'valid')]
        xgb_model = xgb.train(xgb_params, trn_data, num_round, watchlist, early_stopping_rounds=100, verbose_eval=200)
        oof_preds[val_idx] = xgb_model.predict(xgb.DMatrix(train[feats].iloc[val_idx]),
                                               ntree_limit=xgb_model.best_ntree_limit + 50)

        sub_preds += xgb_model.predict(xgb.DMatrix(test[feats]),
                                       ntree_limit=xgb_model.best_ntree_limit + 50) / folds.n_splits
        logger.info('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(val_data, oof_preds[val_idx])))
        del trn_data, val_data, watchlist
        gc.collect()

    logger.info(rmse(oof_preds, target))
    test.loc[:, 'target'] = sub_preds
    test = test.reset_index()
    test[['card_id', 'target']].to_csv('./reports/submission_xgb.csv', index=False)


def read_merged():
    df = pd.read_pickle('./data/interim/merged.pkl')
    return df


def main():
    """ Build lgb model.
    """
    logger = logging.getLogger(__name__)
    logger.info('build lgb model')

    train_df = read_train()
    test_df = read_test()
    kfold_xgboost(train_df, test_df, num_folds=8, stratified=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
