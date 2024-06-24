import pickle

with open(f"models/CNN_20240618_1_history.pkl", "rb") as f:
    history = pickle.load(f)
print(history.history["val_fn"])
