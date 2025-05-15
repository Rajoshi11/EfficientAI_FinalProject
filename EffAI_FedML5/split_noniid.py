import pandas as pd
import numpy as np

# Load full dataset
df = pd.read_csv("data/credit_data.csv")

# Shuffle the dataset for randomness
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define non-IID split rules
client_0 = df[df["age"] < 30]                                     # Young borrowers
client_1 = df[df["duration"] > 24]                                # Long credit duration
client_2 = df[df["amount"] > df["amount"].median()]               # High credit amount
client_3 = df[df["savings"] == "... >= 1000 DM"]                  # High savings
client_4 = df[df["employment_duration"] == "unemployed"]          # Unemployed group

# Remove overlaps
used_indices = set()
for i, client in enumerate([client_0, client_1, client_2, client_3, client_4]):
    current_indices = set(client.index)
    unique_indices = current_indices - used_indices
    used_indices.update(unique_indices)
    locals()[f"client_{i}"] = client.loc[list(unique_indices)]  # FIXED: convert set to list

# Assign remaining data evenly
remaining = df[~df.index.isin(used_indices)]
split_rem = np.array_split(remaining, 5)
for i in range(5):
    locals()[f"client_{i}"] = pd.concat([locals()[f"client_{i}"], split_rem[i]])

# Save each client split
for i in range(5):
    locals()[f"client_{i}"].to_csv(f"client_{i}.csv", index=False)

# Summary
print("Non-IID 5-client split completed.\n")
for i in range(5):
    print(f"Client {i} data shape:", locals()[f"client_{i}"].shape)
