import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(BASE_DIR, "clean"), exist_ok=True)

# ======================================
# Robust CSV Loader
# ======================================
def load_csv_safely(path):

    encodings = ["utf-8", "cp1252", "latin1"]

    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                sep=None,
                engine="python"
            )
            print(f"Loaded with encoding: {enc}")
            return df
        except Exception:
            continue

    raise ValueError(f"Cannot read file: {path}")

# ======================================
# Paths
# ======================================
customer_path = os.path.join(BASE_DIR, "customer_information.csv")
transaction_path = os.path.join(BASE_DIR, "transactions.csv")

print("Customer path:", customer_path)
print("Transaction path:", transaction_path)

# ======================================
# CLEAN CUSTOMER DATA
# ======================================
print("Cleaning customer file...")

df_customers = load_csv_safely(customer_path)

df_customers = df_customers.dropna(subset=["customerID"])
df_customers = df_customers.drop_duplicates(subset=["customerID"])

df_customers.to_csv(
    os.path.join(BASE_DIR, "clean", "customer_information_clean.csv"),
    index=False
)

# ======================================
# CLEAN TRANSACTIONS
# ======================================
print("Cleaning transactions...")

df_transactions = load_csv_safely(transaction_path)

df_transactions = df_transactions.dropna(subset=["customerID"])

df_transactions["timestamp"] = pd.to_datetime(
    df_transactions["timestamp"],
    errors="coerce"
)

df_transactions = df_transactions.dropna(subset=["timestamp"])

# SORT từ cũ → mới
df_transactions = df_transactions.sort_values(
    by="timestamp",
    ascending=True
)

df_transactions.to_csv(
    os.path.join(BASE_DIR, "clean", "transactions_clean.csv"),
    index=False
)

print("DONE. Files saved in /clean")