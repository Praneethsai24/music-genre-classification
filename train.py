import os, json, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dataset import build_manifest, stratified_split
from feature_extraction import extract_vector_from_path

OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG = {
    "duration": 30.0,
    "center_crop": True,
    "test_size": 0.2,
    "random_state": 42,
    "cache_dir": "data/features"
}

def featurize(df, duration, center_crop, cache_dir):
    X = [extract_vector_from_path(p, duration=duration, center_crop=center_crop, cache_dir=cache_dir) for p in df["path"]]
    return np.stack(X, axis=0)

def main():
    df = build_manifest("data/genres")
    tr_df, te_df, tr_csv, te_csv = stratified_split(df, test_size=CONFIG["test_size"], seed=CONFIG["random_state"])
    X_tr = featurize(tr_df, CONFIG["duration"], CONFIG["center_crop"], CONFIG["cache_dir"])
    X_te = featurize(te_df, CONFIG["duration"], CONFIG["center_crop"], CONFIG["cache_dir"])
    le = LabelEncoder()
    y_tr = le.fit_transform(tr_df["label"].values)
    y_te = le.transform(te_df["label"].values)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = Sequential([
        Dense(512, activation="relu", input_shape=(X_tr_s.shape[1],)),
        Dropout(0.4),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(len(le.classes_), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    ckpt_path = os.path.join(OUT_DIR, "best_model.h5")
    cbs = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True)
    ]
    hist = model.fit(X_tr_s, y_tr, validation_data=(X_te_s, y_te), epochs=60, batch_size=32, callbacks=cbs, verbose=2)

    # Always reload the best checkpoint before final eval/save to avoid accidental overwrite
    try:
        from tensorflow.keras.models import load_model
        best = load_model(ckpt_path)
        model = best
    except Exception:
        pass

    # Save artifacts
    model.save(os.path.join(OUT_DIR, "final_model.h5"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.joblib"))
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Basic report for sanity check
    y_pred = np.argmax(model.predict(X_te_s), axis=1)
    report = classification_report(y_te, y_pred, target_names=list(le.classes_), digits=4)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    main()
