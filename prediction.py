import os, json, joblib, numpy as np
import argparse
from tensorflow.keras.models import load_model
from feature_extraction import extract_vector_from_path

def main():
    parser = argparse.ArgumentParser(description="Predict music genre for an audio file.")
    parser.add_argument("--file", required=True, help="Path to audio file (.wav/.mp3/.flac/.au)")
    parser.add_argument("--topk", type=int, default=3, help="Top-K predictions to show")
    args = parser.parse_args()

    path = args.file
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    models_dir = "models"
    model_path = os.path.join(models_dir, "best_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(models_dir, "final_model.h5")
    model = load_model(model_path)
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))
    with open(os.path.join(models_dir, "config.json")) as f:
        cfg = json.load(f)

    x = extract_vector_from_path(path, duration=cfg["duration"], center_crop=cfg["center_crop"], cache_dir=cfg.get("cache_dir")).reshape(1, -1)
    xs = scaler.transform(x)
    prob = model.predict(xs)[0]
    if not np.all(np.isfinite(prob)):
        raise RuntimeError("Model produced non-finite probabilities; check input/scaler/config.")
    topk = np.argsort(prob)[::-1][:max(1, args.topk)]
    for idx in topk:
        print(f"{le.classes_[idx]}: {prob[idx]:.3f}")
    print(f"Top-1: {le.classes_[topk[0]]}")

if __name__ == "__main__":
    main()
