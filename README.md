# Music Genre Classification 

End-to-end music genre classification using Librosa features + Keras MLP. This refactor fixes imports, avoids data leakage, adds feature caching, and aligns CLI + docs with the code.

## Quickstart
1) **Install requirements**
```
pip install -r requirements.txt
```
2) **Download GTZAN**
```
python dataset_download.py
```
3) **Train**
```
python train.py
```
Artifacts saved to `models/`:
- `best_model.h5` (checkpoint)
- `final_model.h5` (final model, best if checkpoint found)
- `scaler.joblib`, `label_encoder.joblib`, `config.json`, `report.txt`

4) **Evaluate (on saved test split, or deterministic stratified split)** 
```
python model_evaluation.py
```

5) **Predict on your own audio**
```
python prediction.py --file path/to/audio.wav --topk 3
```

## Notes
- Features are cached to `data/features/` for fast re-runs.
- Uses MFCC(+delta) + Chroma aggregated over beat-synchronous frames.
- If beat tracking fails, falls back to framewise aggregation.
- GTZAN has known duplicates/mislabels; for research-grade results, use curated folds.

## Requirements
See `requirements.txt`.

