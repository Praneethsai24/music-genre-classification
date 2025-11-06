import os
import hashlib
import librosa
import numpy as np

DEFAULT_SR = 22050  # consistent with many GTZAN
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20

def _cache_path(audio_path: str, duration: float, center_crop: bool, cache_dir: str) -> str:
    """
    Build a deterministic cache filename from audio path and featurization params.
    """
    h = hashlib.sha1(f"{audio_path}|{duration}|{center_crop}|{DEFAULT_SR}|{HOP_LENGTH}|{N_MFCC}".encode()).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(cache_dir, f"{base}.{h}.npy")

def compute_logmels(y, sr, n_mels=N_MELS, hop_length=HOP_LENGTH):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def compute_mfcc_stack(y, sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    return np.vstack([mfcc, mfcc_delta])

def beat_sync_features(y, sr):
    # Harmonic-percussive separation improves beat estimates
    y_harm, y_perc = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr)
    mfcc_stack = compute_mfcc_stack(y, sr)
    # Sync MFCC to beats (sum/mean invariant to varying frame count)
    mfcc_bs = librosa.util.sync(mfcc_stack, beat_frames, aggregate=np.mean)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_bs = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    feats = np.vstack([chroma_bs, mfcc_bs])
    # Aggregate to a fixed-length vector
    mean = np.mean(feats, axis=1)
    std = np.std(feats, axis=1)
    vec = np.hstack([mean, std]).astype(np.float32)
    if not np.all(np.isfinite(vec)):
        # Fallback to global framewise aggregation if beat tracking fails
        mfcc = compute_mfcc_stack(y, sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        feats = np.vstack([chroma, mfcc])
        mean = np.mean(feats, axis=1)
        std = np.std(feats, axis=1)
        vec = np.hstack([mean, std]).astype(np.float32)
    return vec

def extract_vector_from_path(path, duration=30.0, center_crop=True, cache_dir=None):
    """Extract a fixed-length feature vector from an audio file.

    If cache_dir is provided, store/load a cached .npy to avoid re-computation.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cp = _cache_path(path, duration, center_crop, cache_dir)
        if os.path.exists(cp):
            try:
                return np.load(cp)
            except Exception:
                pass  # fall through to recompute

    y, sr = librosa.load(path, sr=DEFAULT_SR, mono=True, duration=None)
    target_len = int(duration * sr)
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad))
    elif center_crop and len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start+target_len]

    vec = beat_sync_features(y, sr)

    if cache_dir:
        try:
            np.save(_cache_path(path, duration, center_crop, cache_dir), vec)
        except Exception:
            pass
    return vec
