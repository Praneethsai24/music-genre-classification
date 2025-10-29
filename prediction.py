import sys
from tensorflow.keras.models import load_model
from feature_extraction import extract_features
from sklearn.preprocessing import StandardScaler
import numpy as np

if len(sys.argv) < 2:
    print('Usage: python src/predict.py file_to_predict.wav')

audio_file = sys.argv[1]
features = extract_features(audio_file).reshape(1, -1)

scaler = StandardScaler() # must load or fit scaler as used in train_model.py
# For demonstration, fit on dummy value - replace with actual scaler from training
features_scaled = scaler.fit_transform(features) 

model = load_model('models/saved_model.h5')
pred = np.argmax(model.predict(features_scaled), axis=1)

# Must load and use label encoder - for demonstration, print integer label
print(f"Predicted genre index: {pred[0]}")
