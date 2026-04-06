# Copilot Instructions: AI Cup 2026 Bird Classification

## Project Context

Multi-class classification of bird radar tracks into 9 categories using gradient boosting models. The dataset contains WKB-encoded spatial trajectories, radar measurements, and temporal data.

## Commands

### Environment Setup
```bash
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Training and Submission
```bash
# Quick baseline model (~2 min, OOF Log Loss: 0.1495)
python run_baseline.py

# Advanced ensemble (~15-30 min, better performance)
python run_pipeline.py
```

### Single Model Testing
```bash
# Test on subset of data
python -c "from src.features import extract_simple_features; import pandas as pd; df = pd.read_csv('data/train.csv').head(100); print(extract_simple_features(df))"
```

## Project Structure

- `data/` - Competition data (train.csv, test.csv, sample_submission.csv)
- `src/features.py` - Feature extraction (trajectory parsing, temporal, radar features)
- `src/train.py` - Ensemble training pipeline (XGBoost + LightGBM + CatBoost)
- `run_baseline.py` - Fast baseline with simple features
- `run_pipeline.py` - Full ensemble with advanced trajectory features
- `outputs/` - Generated submissions and OOF predictions
- `models/` - Serialized trained models

## High-Level Architecture

### Data Flow
1. **Input**: CSV with track_id, WKB trajectories, radar data, timestamps
2. **Feature Engineering**: Extract 40+ features from multiple sources
3. **Training**: Multi-class classification with 5-fold CV
4. **Ensemble**: Average predictions across XGBoost, LightGBM, CatBoost
5. **Output**: Probability predictions for 9 bird categories

### Key Components

**Feature Extraction** (`src/features.py`):
- `parse_trajectory()` - Decodes WKB hex to spatial coordinates
- `extract_trajectory_features()` - Spatial stats, distances, angles
- `extract_temporal_features()` - Time of day, duration, cyclical encoding
- `extract_radar_features()` - Bird size, airspeed, altitude features

**Model Training** (`src/train.py`):
- `train_single_target()` - Train one model for one target class
- `train_ensemble_cv()` - Full 5-fold CV with 3 model types per target
- `predict_test()` - Generate ensemble predictions by averaging all models
- Uses StratifiedKFold to handle class imbalance

## Key Conventions

### Target Classes
Always use this exact order (matches submission format):
```python
TARGET_COLS = ['Clutter', 'Cormorants', 'Pigeons', 'Ducks', 'Geese',
               'Gulls', 'Birds of Prey', 'Waders', 'Songbirds']
```

### Data Handling
- **Train data**: Has `bird_group` and `bird_species` labels
- **Test data**: Missing observer fields (`n_birds_observed`, `observer_comment`, etc.)
- **Feature extraction**: Must handle missing columns gracefully
- **WKB trajectories**: Use `shapely.wkb.loads(hex_string, hex=True)`

### Model Training Pattern
- Each target class gets its own binary classifier
- 5-fold stratified cross-validation
- Three model types per fold: XGBoost, LightGBM, CatBoost
- Final prediction: Average across all folds and model types
- Use `early_stopping_rounds` parameter in model constructor (not fit())

### Evaluation Metric
- **Metric**: Log Loss (lower is better)
- **Target**: Minimize average log loss across all 9 classes
- **Best baseline score**: 0.1495 OOF Log Loss

## Data Characteristics

### Class Imbalance
Highly imbalanced distribution (training data):
- Gulls: 58% (1,503 samples) - dominant class
- Songbirds: 19% (483 samples)
- Minorities: Clutter, Cormorants, Ducks, Geese (<5% each)

**Implication**: Models may overpredict Gulls. Consider class weights or SMOTE.

### Feature Types
- **Trajectory**: WKB-encoded LineString (spatial path of bird)
- **Radar**: `radar_bird_size`, `airspeed`, `min_z`, `max_z` (some have mixed types - convert to numeric)
- **Temporal**: ISO timestamps for start/end of track
- **Metadata**: observer_position, observer_comment (train only)

### Common Data Issues
- Mixed data types in numeric columns (use `pd.to_numeric(..., errors='coerce')`)
- NaN values in radar measurements (fill with 0)
- Test data has fewer columns than train (conditionally handle)

## Development Workflow

1. **Iterate features**: Edit `src/features.py`, run `run_baseline.py` to test quickly
2. **Tune models**: Adjust hyperparameters in `src/train.py`
3. **Validate locally**: Check OOF Log Loss per class to identify weak predictions
4. **Generate submission**: Run pipeline, verify format matches `sample_submission.csv`
5. **Debug**: Use small subset of data for faster iteration

## Performance Notes

- **Baseline (simple features)**: 0.1495 OOF Log Loss
- **Advanced (trajectory parsing)**: Expected ~0.10-0.12 (not yet validated)
- **Hardest classes**: Gulls (0.43), Songbirds (0.30) - high variance in flight patterns
- **Easiest classes**: Clutter (0.04), Ducks (0.06) - distinct signatures

## Submission Format

CSV with columns: `track_id, Clutter, Cormorants, Pigeons, Ducks, Geese, Gulls, Birds of Prey, Waders, Songbirds`
- Probabilities should sum to ~1.0 per row (soft predictions)
- Must have exactly 1,873 rows (1 header + 1,872 test samples)

