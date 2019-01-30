* merge transactions with merchants and train/test
* concat all with test/train marker column, then split
* date to number of days from first
* dummy encoding
* lgbm
- fill nans
- feature combos
- try remove duplicates
```        
# transactions.reset_index(inplace=True)
# transactions['ident'] = transactions.index
# logger.info(f'shape {transactions_merchants.shape!r}')
# transactions_merchants.drop_duplicates(subset=['ident'], keep='last', inplace=True)
# logger.info(f'shape {transactions_merchants.shape!r}')
# logger.info(f'indexes {transactions_merchants.ident.nunique()!r}')
```
- check kernels