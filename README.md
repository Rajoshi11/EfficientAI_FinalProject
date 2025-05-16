# Federated Explainable AI for Privacy-Preserving and Efficient Credit Risk Scoring

---

## Summary

This project develops a **Federated Learning (FL)** pipeline augmented with **Explainable AI (XAI)** for **credit risk scoring** — a key real-world problem in the financial industry. The idea is to collaboratively train machine learning models **without sharing raw data**, thereby preserving privacy while enabling shared intelligence among simulated financial institutions (clients).

We explored this via two phases:

1. **WebSocket-based distributed simulation** to test the communication logic, convergence, and centralized explainability.
2. **Federated Learning with Flower** and **SHAP-based explainability**, simulating decentralized learning and privacy-preserving interpretation across clients.

---

## Objectives

- Develop an **interpretable federated learning system** for credit risk classification.
- Enable **privacy-preserving insights** via SHAP without leaking raw data or labels.
- Evaluate performance trade-offs in **3-client vs. 5-client** setups.
- Quantify the **impact of federated pruning** and **model compression** on explainability.

---

## Research Questions

- How effective are SHAP explanations in a decentralized federated setup?
- Can local SHAP values be aggregated without compromising privacy or fidelity?
- How does the accuracy/interpretability vary across 3-client and 5-client non-IID splits?
- What role does model compression (e.g., pruning) play in interpretability?

---

## Project Structure
```
FEDMLEffAI/
│
├── EffAI_FedML/ # 3-client Flower-based FL + SHAP
│ ├── data/
│ │ ├── client_0.csv
│ │ ├── client_1.csv
│ │ └── client_2.csv
│ ├── client.py
│ ├── server.py
│ ├── plot_metrics.py
│ ├── save_encoder_scaler.py
│ ├── split_noniid.py
│ ├── encoder.pkl / scaler.pkl
│ ├── fed_history.pkl / shap_values.pkl
│ ├── shap_client_0.json ... shap_client_2.json
│ ├── shap_summary_plot.png
│ ├── shap_comparison_clients.png
│ ├── federated_metrics.png
│ ├── global_metrics_over_rounds.csv
│ └── federated_metrics_all.png

├── EffAI_FedML5/ # 5-client Flower-based FL + SHAP
│ ├── data/
│ │ ├── client_0.csv ... client_4.csv
│ ├── client.py
│ ├── server.py
│ ├── plot_metrics.py
│ ├── save_encoder_scaler.py
│ ├── split_noniid.py
│ ├── shap_client_0.json ... shap_client_4.json
│ ├── shap_summary_plot.png
│ ├── shap_comparison_clients.png
│ ├── global_metrics_over_rounds.csv
│ ├── federated_metrics.png
│ └── fed_history.pkl

├── EffAI_WebSockets/ # WebSocket-based DNN FL prototype
│ ├── client.py / server.py
│ ├── data/
│ │ └── credit_data.csv
│ ├── model/
│ │ └── keras_dnn.py
│ ├── plot_metrics.py
│ ├── global_metrics.json
│ ├── global_accuracy.png / global_loss.png

├── README.md # This file
├── requirements.txt
└── .gitignore
```


---

## Phase 1: WebSocket-Based FL Simulation

### Overview
- Designed to test **FedAvg** strategy in a minimal setup.
- Built using raw `socket` programming (TCP).
- 3 clients trained on **non-IID partitions**.

### Flow

1. Load and split the South German Credit dataset.
2. Each client trains a lightweight 3-layer DNN.
3. Server performs weight aggregation.
4. Results are plotted and saved locally.

## Phase 2: Flower-Based Federated Learning with SHAP

### Tools
- `TensorFlow` + `Flower (flwr)`
- `SHAP` for XAI
- `matplotlib` + `pandas` for plots
- 3-layer MLP (32-16-1) with sigmoid output

---

## Data Preprocessing

- Mapped `credit_risk` as binary: **good = 1**, **bad = 0**
- Applied `OneHotEncoder` for categorical features
- Applied `StandardScaler` to numerical features
- Saved `encoder.pkl` and `scaler.pkl` for client-side use
- Partitioned the dataset **non-IID** using criteria like:
  - Age
  - Credit duration
  - Loan amount
  - Installment rate
  - Residence length

---

## Model Architecture

- Lightweight MLP for **edge efficiency**
- Layer config: `[Dense(32) → Dense(16) → Dense(1)]`
- Activation: `ReLU` + `Sigmoid`
- Loss: `Binary Crossentropy`
- Optimizer: `Adam` (client), `SGD` (server)

---

## SHAP Explainability (Client + Server)

- Used `shap.Explainer` on both local and global models
- Shared only **mean SHAP vectors**, not raw features or weights
- Visualized SHAP summaries and client-wise comparison
- Verified interpretability retention post-pruning

---

## Results

### Global Metrics (3 Clients)

| Round | Accuracy | Loss  |
|-------|----------|-------|
| 1     | 0.75     | 0.498 |
| 3     | 0.84     | 0.356 |
| 5     | 0.88     | 0.274 |
| 9     | 0.92     | 0.224 |

Accuracy/Loss - 3 Clients

![WhatsApp Image 2025-05-15 at 19 53 10_65a78aaa](https://github.com/user-attachments/assets/7e76d465-00d5-4f4b-a9b8-aab950b7b48a)

SHAP Summary - 3 Clients

![WhatsApp Image 2025-05-15 at 20 57 26_068d3f56](https://github.com/user-attachments/assets/71eb5a8a-b826-42ce-91b3-3e030cbe7a02)

SHAP Comparision - 3 Clients

![WhatsApp Image 2025-05-14 at 14 05 42_cbd97979](https://github.com/user-attachments/assets/1b7c1d45-5f94-4beb-a543-1bd2b9f6f874)

---

### 📈 Global Metrics (5 Clients)

| Round | Accuracy | Loss  |
|-------|----------|-------|
| 1     | 0.69     | 0.581 |
| 3     | 0.81     | 0.416 |
| 5     | 0.88     | 0.297 |
| 9     | 0.91     | 0.225 |

Accuracy/Loss - 5 Clients

![WhatsApp Image 2025-05-15 at 19 52 16_58fa1001](https://github.com/user-attachments/assets/b154a13c-5986-41fa-aa5d-9d057aa2ea8c)

SHAP Summary - 5 clients

![WhatsApp Image 2025-05-15 at 19 51 55_2135a0d6](https://github.com/user-attachments/assets/3444f864-771f-44b4-9632-ba2fd8dd2196)

SHAP Comparision - 5 Clients

![WhatsApp Image 2025-05-15 at 19 50 23_dee15a8b](https://github.com/user-attachments/assets/4368f69f-282c-4fd9-891e-294a0bed74cc)

---

## System Architecture

System Design

![eff](https://github.com/user-attachments/assets/67149b8e-3f3a-412c-af99-5d91abfd2d16)

---

## Team Responsibilities

| Member            | Role |
|-------------------|------|
| **Rujuta Joshi**  | FL orchestration, SHAP pipeline, client pruning |
| **Greeshma Hedvikar** | Data processing, client-side training |
| **Lavanya Deole** | SHAP visualization, SHAP JSON extraction |
| **Isha Harish**   | Quantization logic, analysis, model efficiency |

---

## How to Run

### WebSocket Prototype

```bash
cd EffAI_WebSockets
python server.py            # Terminal 1
python client.py 0          # Terminal 2
python client.py 1
python client.py 2
```

### Flower-Based Federated Learning (3 Clients)
```bash
cd EffAI_FedML
python server.py
python client.py 0
python client.py 1
python client.py 2
python plot_metrics.py
```

### Flower-Based Federated Learning (5 Clients)
```bash
cd EffAI_FedML5
python server.py
python client.py 0
python client.py 1
python client.py 2
python client.py 3
python client.py 4
python plot_metrics.py
```

## Future Work

Fairness and bias detection in explanations

Streamlit dashboard for live FL monitoring

Edge-device deployment (Jetson Nano, RPi)

Fine-tuning with LoRA / QLoRA or distillation

