# AI Cup 2026 Bird Classification

Competition-ready solution for bird species classification from radar track data.

## 📊 Problem Description

Multi-class classification of bird radar tracks into 9 categories:
- Clutter
- Cormorants
- Pigeons
- Ducks
- Geese
- Gulls
- Birds of Prey
- Waders
- Songbirds

## 🚀 Quick Start

### Setup Environment

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies (already done)
pip install -r requirements.txt
```

### Run Baseline Model

```bash
# Fast baseline model (~ 2 minutes)
python run_baseline.py
```

**Output:** `outputs/baseline_submission.csv`
**Performance:** Average OOF Log Loss: 0.1495

### Run Advanced Ensemble (Optional)

```bash
# Full pipeline with advanced feature engineering (~ 15-30 minutes)
python run_pipeline.py
```

**Output:** `outputs/submission.csv`

## 📁 Project Structure

```
.
├── data/                       # Competition data
│   ├── train.csv              # Training data (2,601 tracks)
│   ├── test.csv               # Test data (1,872 tracks)
│   └── sample_submission.csv  # Submission format
├── src/                       # Source code modules
│   ├── features.py           # Feature engineering (trajectory parsing, temporal, radar)
│   └── train.py              # Ensemble training (XGBoost, LightGBM, CatBoost)
├── outputs/                   # Model outputs and submissions
│   └── baseline_submission.csv
├── models/                    # Saved model checkpoints
├── run_baseline.py           # Quick baseline model script
├── run_pipeline.py           # Full ensemble pipeline
└── requirements.txt          # Python dependencies

```

## 🔧 Features

### Baseline Model Features
- **Radar signatures**: bird size, airspeed, altitude (min/max/range)
- **Temporal**: duration, hour of day, day of week, month, cyclical encoding
- **Trajectory**: approximate length from WKB hex string
- **Observer**: number of birds observed

### Advanced Features (run_pipeline.py)
- **Trajectory parsing**: WKB geometry decoding for spatial coordinates
- **Spatial statistics**: mean, std, min, max for x, y, z coordinates
- **Movement patterns**: total distance, straightness, step distances
- **Directional features**: angle changes, turning behavior
- **Enhanced temporal**: time period categorization (morning/afternoon/evening/night)

## 🎯 Models

### Baseline
- **Algorithm**: XGBoost
- **Training**: 3-fold cross-validation
- **Speed**: ~2 minutes
- **Score**: 0.1495 OOF Log Loss

### Advanced Ensemble
- **Algorithms**: XGBoost + LightGBM + CatBoost
- **Training**: 5-fold cross-validation per model
- **Ensemble**: Average predictions across all models and folds
- **Speed**: ~15-30 minutes
- **Expected improvement**: Lower log loss through ensemble diversity

## 📈 Results

### Per-Class Performance (Baseline)

| Class | OOF Log Loss |
|-------|--------------|
| Clutter | 0.0398 |
| Cormorants | 0.0713 |
| Pigeons | 0.0802 |
| Ducks | 0.0629 |
| Geese | 0.1086 |
| Gulls | 0.4336 |
| Birds of Prey | 0.1235 |
| Waders | 0.1228 |
| Songbirds | 0.3030 |
| **Average** | **0.1495** |

**Insights:**
- Clutter is easiest to predict (low log loss)
- Gulls and Songbirds are most challenging (high log loss) - likely due to:
  - Large class imbalance (Gulls: 58% of training data)
  - High intra-class variability in flight patterns

## 📝 Usage Examples

### Generate Submission

```bash
# Quick baseline
python run_baseline.py
# → outputs/baseline_submission.csv

# Advanced ensemble
python run_pipeline.py
# → outputs/submission.csv
```

### Load and Analyze Results

```python
import pandas as pd

# Load submission
submission = pd.read_csv('outputs/baseline_submission.csv')

# Check prediction probabilities
print(submission.head())

# Verify probabilities sum (should be close to 1 for each row)
print(submission.iloc[:, 1:].sum(axis=1).describe())
```

## 🔍 Data Insights

- **Training samples**: 2,601 radar tracks
- **Test samples**: 1,872 radar tracks
- **Class distribution**: Highly imbalanced
  - Gulls: 58% (1,503 samples)
  - Songbirds: 19% (483 samples)
  - Pigeons, Waders, Birds of Prey: 3-5% each
  - Clutter, Geese, Ducks, Cormorants: <3% each

- **Feature types**:
  - WKB-encoded spatial trajectories
  - Radar measurements (size, speed, altitude)
  - Temporal metadata (timestamps)
  - Observer annotations (train only)

## 🚧 Future Improvements

1. **Class imbalance handling**:
   - SMOTE or class weights
   - Stratified sampling

2. **Feature engineering**:
   - Fourier transforms of trajectories
   - Statistical moments of movement patterns
   - Weather/environmental features if available

3. **Model enhancements**:
   - Neural networks for trajectory sequences
   - Attention mechanisms for temporal patterns
   - Stacking with meta-learners

4. **Ensemble optimization**:
   - Weighted averaging based on validation performance
   - Stacking ensemble (train meta-model on OOF predictions)

## 📄 License

Competition project for educational purposes.
