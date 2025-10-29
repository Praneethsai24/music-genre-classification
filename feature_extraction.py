import librosa
import numpy as np
import os

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return np.hstack([mfcc, chroma, tempo])

def process_dataset(data_dir):
    genres = os.listdir(data_dir)
    X, y = [], []
    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        if os.path.isdir(genre_dir):
            for file in os.listdir(genre_dir):
                if file.endswith('.wav'):
                    features = extract_features(os.path.join(genre_dir, file))
                    X.append(features)
                    y.append(genre)
    return np.array(X), np.array(y)