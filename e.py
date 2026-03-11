import pandas as pd

enriched = pd.read_csv("data/output/enriched_trades_train.csv", parse_dates=["timestamp"])

no_match = enriched[enriched["market_price"].isna()]
print("Day of week distribution của unmatched trades:")
print(no_match["timestamp"].dt.day_name().value_counts())
print("\nSample:")
print(no_match[["investor_id","asset_id","timestamp","side"]].head(10))