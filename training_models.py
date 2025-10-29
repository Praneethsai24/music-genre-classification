import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from feature_extraction import process_dataset

X, y = process_dataset('data/genres')
le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation="relu"),
    Dense(64, activation="relu"),
    Dense(len(np.unique(y)), activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=35, batch_size=16, validation_data=(X_test, y_test))
model.save('models/saved_model.h5')
