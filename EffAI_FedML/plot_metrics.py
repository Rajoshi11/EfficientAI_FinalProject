# import pickle
# import matplotlib.pyplot as plt
# import shap

# # ---------- Federated Metrics ----------
# with open("fed_history.pkl", "rb") as f:
#     history = pickle.load(f)

# loss_values = [l[1] for l in history.losses_distributed]
# accuracy_values = [a[1] for a in history.metrics_distributed["accuracy"]]
# rounds = list(range(1, len(loss_values) + 1))

# # ---------- SHAP Values ----------
# with open("shap_values.pkl", "rb") as f:
#     shap_values, shap_X, feature_names = pickle.load(f)

# # ---------- Plotting ----------
# plt.figure(figsize=(12, 5))

# # Accuracy
# plt.subplot(1, 2, 1)
# plt.plot(rounds, accuracy_values, marker='o')
# plt.title("Federated Accuracy per Round")
# plt.xlabel("Round")
# plt.ylabel("Accuracy")
# plt.grid(True)

# # Loss
# plt.subplot(1, 2, 2)
# plt.plot(rounds, loss_values, marker='o', color='red')
# plt.title("Federated Loss per Round")
# plt.xlabel("Round")
# plt.ylabel("Loss")
# plt.grid(True)

# plt.tight_layout()
# plt.savefig("federated_metrics.png")
# print("Saved federated_metrics.png")

# # ---------- SHAP Summary Plot ----------
# shap.summary_plot(shap_values, features=shap_X, feature_names=feature_names, show=False)
# plt.savefig("shap_summary_plot.png")
# print("Saved shap_summary_plot.png")
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import shap

# ----------------------- Load Training History -----------------------
with open("fed_history.pkl", "rb") as f:
    history = pickle.load(f)

rounds = list(range(1, len(history.losses_distributed) + 1))
loss_values = [l[1] for l in history.losses_distributed]
accuracy_values = [a[1] for a in history.metrics_distributed["accuracy"]]
model_size_values = [s[1] for s in history.metrics_distributed.get("model_size_kb", [(r, 0.0) for r in rounds])]
shap_sim_values = [s[1] for s in history.metrics_distributed.get("shap_cosine_similarity", [(r, 0.0) for r in rounds])]

# ----------------------- Save CSV -----------------------
df_metrics = pd.DataFrame({
    "round": rounds,
    "loss": loss_values,
    "accuracy": accuracy_values,
    "model_size_kb": model_size_values,
    "shap_cosine_similarity": shap_sim_values
})
df_metrics.to_csv("global_metrics_over_rounds.csv", index=False)
print("Saved global_metrics_over_rounds.csv")

# ----------------------- Metric Plots -----------------------
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.plot(rounds, accuracy_values, marker="o")
plt.title("Federated Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(rounds, loss_values, marker="o", color="red")
plt.title("Federated Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(rounds, model_size_values, marker="o", color="green")
plt.title("Model Size (KB) per Round")
plt.xlabel("Round")
plt.ylabel("Size (KB)")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(rounds, shap_sim_values, marker="o", color="purple")
plt.title("SHAP Cosine Similarity per Round")
plt.xlabel("Round")
plt.ylabel("Cosine Similarity")
plt.grid(True)

plt.tight_layout()
plt.savefig("federated_metrics_all.png")
print("Saved federated_metrics_all.png")

# ----------------------- SHAP Summary Plot -----------------------
with open("shap_values.pkl", "rb") as f:
    shap_values, shap_X, feature_names = pickle.load(f)

shap.summary_plot(shap_values, features=shap_X, feature_names=feature_names, show=False)
plt.savefig("shap_summary_plot.png")
print("Saved shap_summary_plot.png")

# ----------------------- SHAP Comparison Across Clients -----------------------
shap_client_0 = [0.2583575343533343, 0.028446289090920854, 0.016305118059526044, 0.006384227675451739, 0.0010223661016090764, 0.03196677510607601, 0.0013464528772436704, 0.01435259591803963, 0.014560228026130544, 0.0006051908909731227, 0.0056047207074472714, 0.004258247139242787, 0.0032810990101541423, 0.0, 0.003987695321285474, 0.008803782644142237, 0.0020339671934333945, 0.02014824289230553, 0.017653805606268957, 0.0, 0.0007553613364618898, 0.0031260530378281475, 0.0, 0.002575439497898334, 0.015718949575869674, 0.0023494914293987683, 0.0038689982509240504, 0.01322123827256728, 0.005363678303104824, 0.001699019029065192, 0.003801533510761027, 0.0033028879861580207, 0.0001332307095251382, 0.0011993003002678346, 0.0034504879977864528, 0.008022852412428857, 0.007054523246351161, 0.0015018717646055552, 0.019157212309865283, 0.003546554180424819, 0.015041818900251143, 0.005093432591630459, 0.0031720824550449237, 0.0009850855475990095, 0.0033710788418694084, 0.0019615201085844704, 0.0030360671460124044, 0.0019429314637556656, 0.0007380455742939401, 0.01399875330159363, 0.0002492649177632635, 0.004366583011019975, 0.00522727865500686, 0.016449475713741658, 0.02514271732557099, 0.014960309405865452, 0.0007253466438346828, 0.0, 0.004654173520680824, 0.005304641264250191, 4.03907493688156e-05, 0.004518301398780508, 0.000998517839851167, 0.0004023261669945593, 0.0011752441761394366, 0.0013119788766140146, 0.00022013336922973296, 0.003657203645546299]
shap_client_1 = [0.23927426919679465, 0.026599054968543354, 0.014658395227588095, 0.006869543937904137, 0.002453711791848761, 0.03596625676561766, 0.0011520711074893687, 0.014030497963866219, 0.01749675875999965, 0.0019374249064053083, 0.010002254896626494, 0.004093635402775058, 0.002944027035233255, 0.002330865261300158, 0.005833586370137828, 0.004226991847405831, 0.0009232745239511113, 0.018187085786058262, 0.0071975699086363094, 0.0002517952529092609, 0.0009189613084308828, 0.008316558244653664, 0.0, 0.003726111494284125, 0.025631100291789815, 0.002539245429743703, 0.0034772918545641, 0.010545293557783593, 0.004167340094440926, 0.00480711540148283, 0.0019687100565526634, 0.003912715516627454, 0.001335423148640742, 0.0006308471358691659, 0.005389176090915374, 0.008349935960924877, 0.008489411921255909, 0.0007698496775701631, 0.007618502563129487, 0.0030980482807072512, 0.007444112890477599, 0.0051137196362794676, 0.0021478095398284464, 0.0023696745142651105, 0.0015465005967145137, 0.001062530737354731, 0.0018068137109434833, 0.001841645667655397, 0.0033431994344573475, 0.01876186540992931, 0.0031530765804151707, 0.000746046915433058, 0.009756325023807588, 0.01736871125667046, 0.020558288827259092, 0.018346738547117762, 0.0023188528489942336, 6.068149727458793e-05, 0.008958021290212247, 0.00706498923652495, 0.0, 0.0006766507940677307, 0.0025915850237167148, 0.0010174912214744865, 0.005488540353905411, 0.008516554078242432, 9.297086036143208e-05, 0.0021904950777379144]
shap_client_2 = [0.1989154639532789, 0.010214296431761855, 0.016336628580908293, 0.010197354462416843, 0.0016982471956095387, 0.027293596040830016, 0.0008875260684949656, 0.008125791770933817, 0.014055068482505165, 0.0002969756787798065, 0.0, 0.002007227301131934, 0.001670085491674641, 0.0007847274034284063, 0.011449817905186985, 0.0039006272082837893, 0.00027285549982140385, 0.010774997062329199, 0.010599445041036236, 0.0, 0.0004685341604364412, 0.0019076881974935517, 0.0002726781435310827, 0.0037911205772931333, 0.016583421862761804, 0.001601293657192341, 0.004142741707433013, 0.01103489787436556, 0.0024953231056841735, 0.005333437227287022, 0.0018294025669960896, 0.0036074493384920035, 0.0008164056124941755, 0.0032937683387504247, 0.0036636545705453754, 0.005175934772581485, 0.0074516562377878745, 0.002431054742913695, 0.0040223968793948505, 0.0019168598358053706, 0.005861628033531208, 0.010163114001275973, 0.0005438471434948356, 0.0020630000999352564, 0.0013149662168075664, 0.0020852098764386033, 0.0008238904951140289, 0.0011338261192043641, 0.0008678893097521092, 0.008967195712433502, 0.001566712415280443, 0.0012077665930148185, 0.005911590394362187, 0.012104450005743029, 0.02088230796103988, 0.011281075770501057, 0.0023420514266937996, 0.0, 0.004015907411742958, 0.006196227180507655, 0.0003009623438119908, 0.0024291614007980874, 0.001856816058947394, 0.0034811950872341802, 0.0011362691797316088, 0.0020515181669189288, 0.00035459609835718843, 0.005361720183398574]

features = list(range(len(shap_client_0)))

plt.figure(figsize=(14, 6))
plt.plot(features, shap_client_0, label="Client 0", marker='o')
plt.plot(features, shap_client_1, label="Client 1", marker='x')
plt.plot(features, shap_client_2, label="Client 2", marker='^')
plt.title("SHAP Values Comparison Across Clients")
plt.xlabel("Feature Index")
plt.ylabel("SHAP Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("shap_comparison_clients.png")
print("Saved shap_comparison_clients.png")
plt.show()
