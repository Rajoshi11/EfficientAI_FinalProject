import pickle
import matplotlib.pyplot as plt
import pandas as pd
import shap
import json
import os

# ----------------------- Load Federated History -----------------------
with open("fed_history.pkl", "rb") as f:
    history = pickle.load(f)

rounds = list(range(1, len(history.losses_distributed) + 1))
loss_values = [l[1] for l in history.losses_distributed]
accuracy_values = [a[1] for a in history.metrics_distributed["accuracy"]]

# ----------------------- Save CSV -----------------------
df_metrics = pd.DataFrame({
    "round": rounds,
    "loss": loss_values,
    "accuracy": accuracy_values
})
df_metrics.to_csv("global_metrics_over_rounds.csv", index=False)
print("Saved global_metrics_over_rounds.csv")

# ----------------------- Accuracy & Loss Plots -----------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rounds, accuracy_values, marker="o")
plt.title("Federated Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(rounds, loss_values, marker="o", color="red")
plt.title("Federated Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("federated_metrics.png")
print("Saved federated_metrics.png")

# ----------------------- SHAP Summary Plot -----------------------
with open("shap_values.pkl", "rb") as f:
    shap_values, shap_X, feature_names = pickle.load(f)

shap.summary_plot(shap_values, features=shap_X, feature_names=feature_names, show=False)
plt.savefig("shap_summary_plot.png")
print("Saved shap_summary_plot.png")

# ----------------------- SHAP Comparison Across Clients -----------------------
shap_vectors = []
num_clients_to_plot = 5  
for i in range(num_clients_to_plot):
    fname = f"shap_client_{i}.json"
    if os.path.exists(fname):
        with open(fname, "r") as f:
            shap_vectors.append(json.load(f))
    else:
        print(f"Missing: {fname}")
        shap_vectors.append([])

# Plot only if SHAP vectors are valid
if all(len(v) > 0 for v in shap_vectors):
    features = list(range(len(shap_vectors[0])))
    plt.figure(figsize=(14, 6))
    markers = ['o', 'x', '^', 's', '*']
    for i, vector in enumerate(shap_vectors):
        plt.plot(features, vector, label=f"Client {i}", marker=markers[i])
    plt.title("SHAP Values Comparison Across Clients")
    plt.xlabel("Feature Index")
    plt.ylabel("SHAP Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("shap_comparison_clients.png")
    print("Saved shap_comparison_clients.png")
    plt.show()
else:
    print("Skipped SHAP comparison plot due to missing vectors.")
