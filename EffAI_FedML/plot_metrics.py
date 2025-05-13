import pickle
import matplotlib.pyplot as plt
import shap

# ---------- Federated Metrics ----------
with open("fed_history.pkl", "rb") as f:
    history = pickle.load(f)

loss_values = [l[1] for l in history.losses_distributed]
accuracy_values = [a[1] for a in history.metrics_distributed["accuracy"]]
rounds = list(range(1, len(loss_values) + 1))

# ---------- SHAP Values ----------
with open("shap_values.pkl", "rb") as f:
    shap_values, shap_X, feature_names = pickle.load(f)

# ---------- Plotting ----------
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(rounds, accuracy_values, marker='o')
plt.title("Federated Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(rounds, loss_values, marker='o', color='red')
plt.title("Federated Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("federated_metrics.png")
print("Saved federated_metrics.png")

# ---------- SHAP Summary Plot ----------
shap.summary_plot(shap_values, features=shap_X, feature_names=feature_names, show=False)
plt.savefig("shap_summary_plot.png")
print("Saved shap_summary_plot.png")
