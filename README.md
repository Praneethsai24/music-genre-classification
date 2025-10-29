# Music Genre Classification Using Neural Networks and Rhythm Patterns

Automated end-to-end project for classifying music genres using deep learning focused on rhythm and timbre features.

## Project Structure
music-genre-classification/
│
├─ README.md
├─ requirements.txt
├─ data/
│   └─ dataset_download.py
├─ src/
│   ├─ feature_extraction.py
│   ├─ training_models.py
│   ├─ model_evaluation.py
│   └─ prediction.py
├─ notebooks/
│   └─ main_genre_classification.ipynb
├─ models/
└─ docs/


## Features
- Data Acquisition (GTZAN via download script)
- Audio Feature Extraction (MFCCs, Chroma, Rhythm)
- Model Training (ML & Deep Learning)
- Evaluation & Visualization
- One-command prediction for your own music
- Model/deployment code

## Usage
1. Install requirements:  
   `pip install -r requirements.txt`
2. Download dataset:  
   `python data/download_dataset.py`
3. Run Jupyter notebook in `notebooks/main_genre_classification.ipynb` OR  
   Train/evaluate from CLI with `src/train_model.py`
4. Make predictions:  
   `python src/predict.py --file path_to_audio.wav`

## Requirements
See `requirements.txt`.

## References
- GTZAN, Librosa, Keras/Tensorflow, Scikit-learn
