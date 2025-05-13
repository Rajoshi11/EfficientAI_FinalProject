import socket
import json
import numpy as np
from tensorflow.keras.utils import to_categorical
from data.data_loader import load_credit_data
from model.keras_dnn import build_model
from tqdm import tqdm
import sys

client_id = int(sys.argv[1])
NUM_ROUNDS = 9
EPOCHS_PER_ROUND = 3
PORT = 1109

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', PORT))
print(f"[+] Connected to server as Client {client_id}")

x_train, y_train, x_test, y_test = load_credit_data(client_id)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

model = build_model(input_dim=x_train.shape[1])

for rnd in tqdm(range(NUM_ROUNDS), desc=f"Client {client_id} Training Rounds"):
    model.fit(x_train, y_train, epochs=EPOCHS_PER_ROUND, batch_size=32, verbose=0, validation_data=(x_test, y_test))
    
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Client {client_id} [Round {rnd+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    payload = {
        "weights": [w.tolist() for w in model.get_weights()],
        "metrics": {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
    }
    serialized = json.dumps(payload) + "END"
    sock.sendall(serialized.encode())

    data = b""
    while True:
        packet = sock.recv(4096)
        if not packet:
            break
        data += packet
        if b"END" in data:
            break

    new_weights = json.loads(data.decode().split("END")[0])
    model.set_weights([np.array(w) for w in new_weights])
    print(f"[Round {rnd+1}] Updated model with global weights")

sock.close()
