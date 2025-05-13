import socket
import threading
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

NUM_CLIENTS = 3
NUM_ROUNDS = 9
PORT = 1109

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('127.0.0.1', PORT))
server.listen(NUM_CLIENTS)
print(f"[*] Server listening on 127.0.0.1:{PORT}")

connections = []
client_weights_per_round = defaultdict(list)
metrics_per_round = defaultdict(lambda: defaultdict(list))
lock = threading.Lock()
barrier = threading.Barrier(NUM_CLIENTS)

def fed_avg(weights_list):
    return [np.mean(layer, axis=0) for layer in zip(*weights_list)]

def handle_client(conn, addr, client_id):
    print(f"[+] Client {client_id} connected from {addr}")
    barrier.wait()
    print(f"Client {client_id} Waiting for round 1 weights...")

    for rnd in range(NUM_ROUNDS):
        try:
            data = b""
            while True:
                packet = conn.recv(4096)
                if not packet:
                    break
                data += packet
                if b"END" in data:
                    break

            payload = json.loads(data.decode().split("END")[0])
            weights = [np.array(w) for w in payload["weights"]]
            metrics = payload["metrics"]

            print(f"[Round {rnd+1}] Received weights from Client {client_id}")
            with lock:
                client_weights_per_round[rnd].append(weights)
                for k in metrics:
                    metrics_per_round[rnd][k].append(metrics[k])

            # Wait for all clients for current round
            while True:
                with lock:
                    if len(client_weights_per_round[rnd]) == NUM_CLIENTS:
                        global_weights = fed_avg(client_weights_per_round[rnd])
                        serialized = json.dumps([w.tolist() for w in global_weights]) + "END"
                        try:
                            conn.sendall(serialized.encode())
                        except:
                            print(f"[!] Failed to send weights to Client {client_id}")
                        break

        except Exception as e:
            print(f"[!] Client {client_id} Error: {e}")
            break

    conn.close()

# Accept connections
for i in range(NUM_CLIENTS):
    conn, addr = server.accept()
    connections.append(conn)
    threading.Thread(target=handle_client, args=(conn, addr, i)).start()

print("[*] All clients connected. Training starts...")

# Wait for all threads to finish
for t in threading.enumerate():
    if t is not threading.main_thread():
        t.join()

# Aggregate metrics
avg_metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
for r in range(NUM_ROUNDS):
    for metric in avg_metrics:
        avg_metrics[metric].append(np.mean(metrics_per_round[r][metric]))

with open("global_metrics.json", "w") as f:
    json.dump(avg_metrics, f, indent=2)

print("[*] Global metrics saved to global_metrics.json")
server.close()
print("[*] All clients disconnected. Server shutting down.")