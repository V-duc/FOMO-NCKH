# import pandas as pd
# tx = pd.read_csv('data/input/transactions.csv')
# print('Raw tx shape:', tx.shape)
# print('Unique transactionID:', tx['transactionID'].nunique())
# print('Duplicate transactionID:', tx['transactionID'].duplicated().sum())

# # Xem 1 transactionID bị duplicate trông như thế nào
# dup_id = tx[tx['transactionID'].duplicated(keep=False)]['transactionID'].iloc[0]
# print('\nSample duplicate transactionID:', dup_id)
# print(tx[tx['transactionID'] == dup_id].to_string())


# import pandas as pd
# tx = pd.read_csv('data/input/transactions.csv')
# print(tx.shape)
# print(tx['transactionID'].nunique())
# print(tx[(tx['timestamp'] >= '2020-07-01') & (tx['timestamp'] <= '2022-11-30')].shape)
import pandas as pd
tx = pd.read_csv('data/input/transactions.csv')
cu = pd.read_csv('data/input/customer_information.csv')

# Filter period
tx['timestamp'] = pd.to_datetime(tx['timestamp'])
tx = tx[(tx['timestamp'] >= '2020-07-01') & (tx['timestamp'] <= '2022-11-30')]

# Filter Stock
assets = pd.read_csv('data/input/asset_information.csv')
stock_isins = assets[assets['assetCategory']=='Stock']['ISIN'].unique()
tx = tx[tx['ISIN'].isin(stock_isins)]

# Filter Mass+Premium
cu_dedup = cu.sort_values('timestamp').drop_duplicates('customerID', keep='last')
mass_premium_ids = cu_dedup[cu_dedup['customerType'].isin(['Mass','Premium'])]['customerID'].unique()
tx_mp = tx[tx['customerID'].isin(mass_premium_ids)]
print("Mass+Premium tx:", tx_mp.shape)
print("Unique investors:", tx_mp['customerID'].nunique())

# Thử thêm filter: chỉ giữ investor có >= 2 transactions?
for min_tx in [2, 3, 5]:
    counts = tx_mp.groupby('customerID').size()
    filtered = tx_mp[tx_mp['customerID'].isin(counts[counts >= min_tx].index)]
    print(f"Min {min_tx} tx: {filtered.shape}, investors: {filtered['customerID'].nunique()}")