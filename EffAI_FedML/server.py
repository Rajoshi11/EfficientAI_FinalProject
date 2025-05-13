import flwr as fl
import numpy as np
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

NUM_ROUNDS = 9

# ---------- Strategy ----------
def weighted_average(metrics):
    total = sum([num_examples for num_examples, _ in metrics])
    loss = sum([num_examples * m["loss"] for num_examples, m in metrics]) / total
    accuracy = sum([num_examples * m["accuracy"] for num_examples, m in metrics]) / total
    return {"loss": loss, "accuracy": accuracy}

strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
)

# ---------- Flower Server ----------
print("[Server] Starting Flower server...\n")
history = fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)

print("\n[Server] Training complete. Computing SHAP values...")

# ---------- SHAP Explainability ----------

# Load encoder and scaler
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load full dataset
df = pd.read_csv("data/credit_data.csv")

# Separate label
y = (df["credit_risk"] == "good").astype(int)

# Prepare categorical and numerical features
cat_cols = df.select_dtypes(include="object").drop(columns=["credit_risk"]).columns
num_cols = scaler.feature_names_in_

# Encode and scale
X_cat = encoder.transform(df[cat_cols])
X_num = scaler.transform(df[num_cols])

X = np.hstack((X_num, X_cat))


# Create the same DNN model
input_dim = X.shape[1]
model = Sequential([
    Dense((input_dim + 1) // 2, input_dim=input_dim, activation="tanh"),
    Dense((input_dim + 1) // 4, activation="tanh"),
    Dense((input_dim + 1) // 8, activation="tanh"),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="SGD", loss="binary_crossentropy", metrics=["accuracy"])

# SHAP kernel explainer
explainer = shap.Explainer(model.predict, X[:100])
shap_values = explainer(X[:100])


# Save history
with open("fed_history.pkl", "wb") as f:
    pickle.dump(history, f)

# Save SHAP values
with open("shap_values.pkl", "wb") as f:
    pickle.dump((shap_values, X[:100], list(num_cols) + list(encoder.get_feature_names_out(cat_cols))), f)
# # Plot summary
# shap.summary_plot(shap_values, features=X[:100], feature_names=list(num_cols) + list(encoder.get_feature_names_out(cat_cols)))
# plt.savefig("shap_summary_plot.png")
# plt.show()
# print("SHAP summary plot saved as shap_summary_plot.png")