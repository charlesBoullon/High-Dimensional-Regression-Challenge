# High-Dimensional Regression Challenge (UP2 Data Science Major 2024–2025)

## Overview
This project, completed within the UP2 Data Science Major (2024–2025), tackles predictive modeling in a high-dimensional regression setting (100 samples × 100 features).  
**Objective:** Compare various regression and ensemble models to generate test set predictions, optimizing for the lowest RMSE.  
The repository demonstrates my ability to handle complex, p ≫ n scenarios and deploy robust machine learning pipelines.

---

## Objectives
- Build a predictive model for a continuous variable `y`
- Compare regression and ensemble techniques
- Address high dimensionality, multicollinearity, feature selection
- Produce a final prediction file (`BOULLON_Charles.txt`)
- Use RMSE as the performance metric

---

## Machine Learning Skills Demonstrated
- High-dimensional regression
- Regularization: Ridge & LASSO
- Feature selection (LASSO, Stepwise AIC)
- Dimension reduction (PCR, PLS)
- Ensemble learning (Random Forest, Gradient Boosting)
- Hyperparameter tuning & cross-validation
- Data preprocessing and train/test split
- Model comparison with RMSE
- Clean, reproducible R code

---

## Dataset
- `data.txt`: 100 training samples, 1 target + 100 predictors
- `Xtest.txt`: 100 × 100 test matrix  
**Goal:** Generate 100 predictions (one per test row)

---

## Methodology Summary

### 1. Preprocessing
- Extraction of `y` and predictors `X`
- Standardization for penalized models
- Baseline RMSE (mean predictor)

### 2. Models Compared *(10-fold cross-validation)*
| Model               | Purpose                            |
|---------------------|------------------------------------|
| Ridge Regression    | Penalization, multicollinearity    |
| LASSO               | Selection + shrinkage              |
| PCR                 | PCA-based regression               |
| PLS                 | Latent variable regression         |
| Stepwise Regression | AIC-based model selection          |
| Gradient Boosting   | Nonlinear ensemble                 |
| Random Forest       | Bagging-based ensemble             |

Full RMSE comparisons are included in the script.

### Best Model: LASSO Pre-Selection + Stepwise Regression
- LASSO for variable selection
- Stepwise regression (AIC) for refinement
- Achieved lowest RMSE (~2.48 on validation set)
- Used for final predictions

### 3. Final Predictions
Steps applied on `Xtest.txt`:
- Standardize with training params
- Select LASSO-chosen features
- Predict with Stepwise model
- Output in `.txt` file

**Final submissions:**
- `BOULLON_Charles.txt` : Stepwise Regression model
- `BOULLON_Charles2.txt` (optional): Random Forest model

---

## 📁 Repository Structure
├── README.md
├── challenge_script.R # Full pipeline (preprocessing, modeling, prediction)
├── data.txt # Training set
├── Xtest.txt # Test set
├── BOULLON_Charles.txt # Predictions (Stepwise model)
└── BOULLON_Charles2.txt # Optional predictions (Random Forest)


---

## 🛠️ Technologies
- R
- glmnet
- caret
- pls
- MASS
- randomForest
- gbm

---

## Contact
**Interested in my work or want to connect about data science topics? Reach out anytime!**


