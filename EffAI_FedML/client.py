# import flwr as fl
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# import pickle
# import sys

# # Load pre-saved encoder and scaler
# with open("encoder.pkl", "rb") as f:
#     encoder = pickle.load(f)

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# def load_client_data(client_id):
#     df = pd.read_csv(f"client_{client_id}.csv")

#     # One-hot encode labels
#     df["credit_risk"] = df["credit_risk"].map({"good": 1, "bad": 0})

#     cat_cols = encoder.feature_names_in_
#     num_cols = scaler.feature_names_in_

#     X_cat = encoder.transform(df[cat_cols])
#     X_num = scaler.transform(df[num_cols])
#     X = np.hstack((X_num, X_cat))
#     y = df["credit_risk"].values

#     return train_test_split(X, y, test_size=0.2, random_state=42)

# def build_model(input_dim):
#     model = Sequential([
#         Dense(32, activation="relu", input_shape=(input_dim,)),
#         Dense(16, activation="relu"),
#         Dense(1, activation="sigmoid")
#     ])
#     model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
#     return model

# # Load data and model
# client_id = sys.argv[1]
# x_train, x_test, y_train, y_test = load_client_data(client_id)
# model = build_model(x_train.shape[1])

# # Flower client
# class FlowerClient(fl.client.NumPyClient):
#     def get_parameters(self, config): return model.get_weights()
#     def set_parameters(self, parameters): model.set_weights(parameters)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1)
#         return model.get_weights(), len(x_train), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
#         return float(loss), len(x_test), {
#             "loss": float(loss),
#             "accuracy": float(accuracy)
#         }

# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
import flwr as fl
import pandas as pd
import numpy as np
import shap
import pickle
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow_model_optimization.sparsity.keras import (
    prune_low_magnitude, PolynomialDecay, UpdatePruningStep, strip_pruning
)

# Load encoder and scaler
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def load_client_data(client_id):
    df = pd.read_csv(f"client_{client_id}.csv")
    df["credit_risk"] = df["credit_risk"].map({"good": 1, "bad": 0})
    cat_cols = encoder.feature_names_in_
    num_cols = scaler.feature_names_in_
    X_cat = encoder.transform(df[cat_cols])
    X_num = scaler.transform(df[num_cols])
    X = np.hstack((X_num, X_cat))
    y = df["credit_risk"].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_dim):
    pruning_params = {
        "pruning_schedule": PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=1000,
            frequency=100
        )
    }
    model = Sequential([
        prune_low_magnitude(Dense(32, activation="relu"), **pruning_params),
        prune_low_magnitude(Dense(16, activation="relu"), **pruning_params),
        prune_low_magnitude(Dense(1, activation="sigmoid"), **pruning_params)
    ])
    model.build(input_shape=(None, input_dim))
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

client_id = sys.argv[1]
x_train, x_test, y_train, y_test = load_client_data(client_id)
model = build_model(x_train.shape[1])
callbacks = [UpdatePruningStep()]

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.prev_shap_vector = None

    def get_parameters(self, config):
        return model.get_weights()

    def set_parameters(self, parameters):
        model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0, callbacks=callbacks)

        # Strip pruning for size measurement
        stripped_model = strip_pruning(model)
        size_bytes = stripped_model.count_params() * 4  # 4 bytes per float32
        size_kb = size_bytes / 1024

        # SHAP
        explainer = shap.Explainer(model.predict, x_train[:100])
        shap_vals = explainer(x_train[:100])
        curr_vector = np.abs(shap_vals.values).mean(axis=0)

        if self.prev_shap_vector is not None:
            cosine_sim = float(
                np.dot(self.prev_shap_vector, curr_vector) /
                (np.linalg.norm(self.prev_shap_vector) * np.linalg.norm(curr_vector))
            )
        else:
            cosine_sim = 0.0

        self.prev_shap_vector = curr_vector

        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        return model.get_weights(), len(x_train), {
            "loss": float(loss),
            "accuracy": float(acc),
            "model_size_kb": size_kb,
            "shap_cosine_similarity": cosine_sim
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        return float(loss), len(x_test), {
            "loss": float(loss),
            "accuracy": float(acc)
        }

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
