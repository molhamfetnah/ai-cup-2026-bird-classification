# 🚀 Quick Start Guide

Get your first submission in under 5 minutes!

## Step 1: Activate Environment
```bash
source .venv/bin/activate
```

## Step 2: Run Baseline Model
```bash
python run_baseline.py
```

**Expected output:**
```
Baseline Model Pipeline
============================================================
Train: (2601, 16), Test: (1872, 9)
Creating target columns...
Extracting simple features...

Training Clutter...
Clutter OOF Log Loss: 0.039751
...
Average OOF Log Loss: 0.149529

Generating test predictions...
============================================================
Baseline submission saved!
============================================================
```

## Step 3: Verify Submission
```bash
./verify_submission.sh
```

**Expected output:**
```
==========================================
✅ Submission is valid!
==========================================
File: outputs/baseline_submission.csv
Ready to submit! 🚀
```

## Step 4: Submit to Competition

Your submission file is ready at: **`outputs/baseline_submission.csv`**

Upload this file to the competition platform.

---

## What's Next?

### Improve Your Score

Try the advanced ensemble model for better performance:
```bash
python run_pipeline.py  # ~15-30 minutes
```

### Understand Your Features
```bash
less FEATURES.md  # Feature documentation
```

### Analyze Results
```python
import pandas as pd

# Load submission
sub = pd.read_csv('outputs/baseline_submission.csv')

# Check which class gets highest predictions
print(sub.iloc[:, 1:].idxmax(axis=1).value_counts())

# Compare with training distribution
train = pd.read_csv('data/train.csv')
print(train['bird_group'].value_counts())
```

### Debug Issues

**Problem**: Import errors
```bash
pip install -r requirements.txt
```

**Problem**: Data not found
```bash
ls data/  # Should show train.csv, test.csv, sample_submission.csv
```

**Problem**: Slow execution
```bash
# Use smaller subset for testing
python -c "
import pandas as pd
from run_baseline import extract_simple_features, create_target_columns

train = pd.read_csv('data/train.csv').head(100)
train = create_target_columns(train)
features = extract_simple_features(train)
print(features.head())
"
```

## Expected Timeline

| Task | Duration | Output |
|------|----------|--------|
| Setup environment | 2 min | Virtual env activated |
| Run baseline | 2 min | baseline_submission.csv |
| Verify submission | 5 sec | Validation passed |
| **Total** | **~5 min** | **Ready to submit** |

## Performance Benchmarks

| Model | OOF Log Loss | Time | Features |
|-------|--------------|------|----------|
| Baseline (XGBoost) | 0.1495 | 2 min | 14 simple |
| Advanced Ensemble | ~0.12 (est.) | 15-30 min | 40+ complex |

---

## Troubleshooting

### Common Issues

**1. Module not found**
```bash
source .venv/bin/activate  # Make sure venv is active
pip list | grep -E "pandas|xgboost"  # Check packages installed
```

**2. Memory error**
```bash
# Reduce data size for testing
python -c "import pandas as pd; pd.read_csv('data/train.csv').head(500).to_csv('data/train_small.csv')"
# Then edit run_baseline.py to use train_small.csv
```

**3. Wrong submission format**
```bash
# Check your submission matches expected format
head -1 outputs/baseline_submission.csv
head -1 data/sample_submission.csv  # Should be identical
```

### Getting Help

1. Check README.md for detailed documentation
2. Review FEATURES.md for feature engineering details
3. Look at .github/copilot-instructions.md for architecture overview

---

## Tips for Improvement

1. **Feature Engineering**: Add interaction features (e.g., `airspeed × altitude_range`)
2. **Hyperparameter Tuning**: Adjust `max_depth`, `learning_rate` in model configs
3. **Class Balancing**: Add class weights to handle Gulls dominance
4. **Ensemble Weights**: Weight models based on per-class performance
5. **Cross-Validation**: Increase folds from 3 to 5 for more robust estimates

Good luck! 🎯
