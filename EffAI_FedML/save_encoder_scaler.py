import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# Load dataset
df = pd.read_csv("data/credit_data.csv")

# Separate label
if "credit_risk" in df.columns:
    df = df.drop(columns=["credit_risk"])  # drop label column if present

# Identify column types
categorical_cols = df.select_dtypes(include="object").columns.tolist()
numerical_cols = df.select_dtypes(exclude="object").columns.tolist()

# Fit encoder and scaler
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(df[categorical_cols])

scaler = StandardScaler()
scaler.fit(df[numerical_cols])

# Save encoder and scaler
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("encoder.pkl and scaler.pkl saved successfully.")
print("Encoder categories:", encoder.categories_)
print("Scaler mean:", scaler.mean_)
print("Scaler scale:", scaler.scale_)
print("Scaler var:", scaler.var_)
print("Scaler var shape:", scaler.var_.shape)