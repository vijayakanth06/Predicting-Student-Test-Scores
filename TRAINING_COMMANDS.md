# Final Training Commands

## Quick Start (Recommended)

**Full power mode** - All optimizations enabled:
```powershell
conda activate gpu-env
python train_final_top3.py
```

**Expected**: 8.50-8.58 CV → 8.47-8.55 LB (Top 3!)
**Time**: ~8-10 hours (with GPU)

---

## Advanced Options

### Without Pseudo-Labeling (faster, ~4 hours)
```powershell
python train_final_top3.py --no-pseudo
```

### Without Hill Climbing (faster optimization)
```powershell
python train_final_top3.py --no-hill-climb
```

### Minimal (fastest, ~4 hours)
```powershell
python train_final_top3.py --no-pseudo --no-hill-climb
```

---

## What's Included

**6 Diverse Models:**
1. LightGBM (GPU) - Best performer
2. CatBoost (CPU) - Strong ensemble member
3. XGBoost (GPU) - Proven winner
4. ExtraTrees - Different algorithm (sklearn)
5. HistGradientBoosting - Fast sklearn GBDT
6. Ridge - Linear baseline

**Techniques:**
- ✅ 10-Fold CV for stability
- ✅ 56 engineered features
- ✅ 5000 iterations (main models)
- ✅ Pseudo-labeling (Chris Deotte's method)
- ✅ Hill climbing ensemble optimization

---

## Expected Performance

**CV → LB Gap**: ~0.035 RMSE

| Config | CV Score | Expected LB | Top 3? |
|--------|----------|-------------|--------|
| Full (recommended) | 8.50-8.58 | 8.47-8.55 | ✅ |
| No pseudo | 8.65-8.70 | 8.62-8.67 | ⚠️ |
| Minimal | 8.72-8.75 | 8.68-8.72 | ❌ |

---

## Troubleshooting

**"ModuleNotFoundError"**:
- Solution: Run `conda activate gpu-env` first

**Out of memory**:
- Reduce n_folds in config.py from 10 to 5
- Or comment out Ridge/ExtraTrees/HistGB (keep top 3 models)

**Too slow**:
- Use `--no-pseudo` flag (cuts time in half)
- Reduce iterations in config.py

---

## After Training

1. Check the generated submission file in `submissions/`
2. Upload to Kaggle
3. Wait for LB score
4. If < 8.54 → 🎉 Top 3!
5. If 8.54-8.60 → Try one more iteration
6. If > 8.60 → Review techniques

---

**Good luck reaching Top 3!** 🚀
