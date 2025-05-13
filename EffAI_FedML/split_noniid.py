import pandas as pd
import numpy as np

# Load full dataset
df = pd.read_csv("data/credit_data.csv")

# Shuffle the dataset first
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define client splits (non-IID based on different distributions)
client_0 = df[df["age"] < 30]                      # Younger age group
client_1 = df[(df["duration"] > 24)]               # Longer credit duration
client_2 = df[(df["amount"] > df["amount"].median())]  # Larger credit amounts

# Avoid overlap
used_indices = set(client_0.index).union(set(client_1.index))
client_1 = client_1[~client_1.index.isin(client_0.index)]
used_indices.update(client_1.index)
client_2 = client_2[~client_2.index.isin(used_indices)]

# Remaining entries if any, split evenly across the three
remaining = df[~df.index.isin(client_0.index.union(client_1.index).union(client_2.index))]
split_rem = np.array_split(remaining, 3)
client_0 = pd.concat([client_0, split_rem[0]])
client_1 = pd.concat([client_1, split_rem[1]])
client_2 = pd.concat([client_2, split_rem[2]])

# Save to CSVs
client_0.to_csv("client_0.csv", index=False)
client_1.to_csv("client_1.csv", index=False)
client_2.to_csv("client_2.csv", index=False)

print("✅ Non-IID data split saved for client_0.csv, client_1.csv, client_2.csv")
print("Client 0 data shape:", client_0.shape)
print("Client 1 data shape:", client_1.shape)
print("Client 2 data shape:", client_2.shape)