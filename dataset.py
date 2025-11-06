import os, glob, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

AUDIO_EXTS = (".wav", ".au", ".flac", ".mp3")

def build_manifest(root="data/genres"):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    rows = []
    for genre in sorted(os.listdir(root)):
        gdir = os.path.join(root, genre)
        if not os.path.isdir(gdir):
            continue
        files = []
        for ext in AUDIO_EXTS:
            files.extend(glob.glob(os.path.join(gdir, f"*{ext}")))
        for path in sorted(files):
            rows.append({"path": path, "label": genre})
    if not rows:
        raise RuntimeError(f"No audio files found under {root}. Expected subfolders per genre.")
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)

def stratified_split(df, test_size=0.2, seed=42, out_dir="data/splits"):
    os.makedirs(out_dir, exist_ok=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    y = df["label"].values
    idx_tr, idx_te = next(sss.split(df, y))
    train_df, test_df = df.iloc[idx_tr], df.iloc[idx_te]
    train_csv = os.path.join(out_dir, "train.csv")
    test_csv = os.path.join(out_dir, "test.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    return train_df, test_df, train_csv, test_csv
