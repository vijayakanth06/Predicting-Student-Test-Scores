# Quick Test All Strategies

**Run these commands one by one:**

## 1. Test Target Transformations (2-3 hours)
```powershell
conda activate gpu-env
python target_transformation.py
```

**Expected output:**
- Tests: none, log, sqrt transformations
- Shows which one gives best CV score
- Auto-saves best submission
- Expected improvement: 0.02-0.05 RMSE

---

## 2. Test Quantile Ensemble (3-4 hours)
```powershell
conda activate gpu-env
python quantile_ensemble.py
```

**Expected output:**
- Tests 3-quantile and 5-quantile ensembles
- Shows improvement over single model
- Auto-saves both submissions
- Expected improvement: 0.02-0.04 RMSE

---

## 3. Combine Best Approaches (After testing above)

After you know which works best, I can create a combined script that uses:
- Best target transformation
- Quantile ensemble
- Hill climbing optimization

---

## What to Copy-Paste for Original Dataset Search

While the scripts run, please copy-paste content from:

**1. Kaggle Data Tab**  
https://www.kaggle.com/competitions/playground-series-s6e1/data
→ Copy the entire "Dataset Description" section

**2. Top Discussions**  
https://www.kaggle.com/competitions/playground-series-s6e1/discussion?sort=votes
→ Copy titles of top 10 discussions

**3. Top Notebooks**  
https://www.kaggle.com/competitions/playground-series-s6e1/code?sortBy=voteCount
→ Copy titles of top 10 notebooks

This will help me find if there's an original dataset to blend!
