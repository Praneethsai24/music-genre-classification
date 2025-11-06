import os, json, joblib, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from dataset import build_manifest, stratified_split
from feature_extraction import extract_vector_from_path

MODELS_DIR = "models"

def main():
    cfg_path = os.path.join(MODELS_DIR, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("models/config.json not found. Train the model first.")
    with open(cfg_path) as f:
        cfg = json.load(f)

    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    model_path = os.path.join(MODELS_DIR, "best_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "final_model.h5")
    model = load_model(model_path)

    import pandas as pd
    split_te = "data/splits/test.csv"
    if os.path.exists(split_te):
        te_df = pd.read_csv(split_te)
    else:
        # Recreate split deterministically to avoid leakage
        df = build_manifest("data/genres")
        _, te_df, _, _ = stratified_split(df, test_size=cfg.get("test_size", 0.2), seed=cfg.get("random_state", 42))

    X = np.stack([extract_vector_from_path(p, duration=cfg["duration"], center_crop=cfg["center_crop"], cache_dir=cfg.get("cache_dir")) for p in te_df["path"]])
    Xs = scaler.transform(X)
    y_true = le.transform(te_df["label"])
    y_prob = model.predict(Xs)
    y_pred = np.argmax(y_prob, axis=1)

    print(classification_report(y_true, y_pred, target_names=list(le.classes_), digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
