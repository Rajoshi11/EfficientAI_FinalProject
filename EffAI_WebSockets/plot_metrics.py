import json
import matplotlib.pyplot as plt

with open("global_metrics.json", "r") as f:
    metrics = json.load(f)

rounds = list(range(1, len(metrics["train_loss"]) + 1))

# Accuracy
plt.figure()
plt.plot(rounds, metrics["train_acc"], label="Train Accuracy")
plt.plot(rounds, metrics["val_acc"], label="Val Accuracy")
plt.title("Global Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("global_accuracy.png")
plt.show()

# Loss
plt.figure()
plt.plot(rounds, metrics["train_loss"], label="Train Loss")
plt.plot(rounds, metrics["val_loss"], label="Val Loss")
plt.title("Global Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("global_loss.png")
plt.show()
