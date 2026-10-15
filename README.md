# Home Credit Default Risk Prediction
Machine learning project to estimate applicant default risk using the Home Credit relational datasets.
# Objective
Predict TARGET (1 = payment difficulty/default risk, 0 = no payment difficulty) and evaluate ranking performance with ROC-AUC and average precision. This is an experimental decision-support project, not a deployed lending decision system.
# Data
Uses application_train.csv, bureau.csv, bureau_balance.csv, credit_card_balance.csv, installments_payments.csv, POS_CASH_balance.csv, and previous_application.csv from the Home Credit Default Risk dataset. Obtain the datasets separately from the original data provider; they are not included in this repository.
# Methods
- Exploratory data analysis, missing-value and anomaly checks
- Customer-level aggregation and relational joins
- Feature engineering and preprocessing
- Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM and CatBoost
- Stratified validation, ROC-AUC, PR-AUC, KS, threshold cost analysis and SHAP
# Repository structure
notebooks/Home_Loan_Default_Risk.ipynb  # Original exploratory workflow (outputs cleared)
src/evaluation.py                      # Reusable probability-based metrics and threshold tuning
requirements.txt
.gitignore
# Setup
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
Open notebooks/Home_Loan_Default_Risk.ipynb. Before running, change its Colab-specific Google Drive mount and data_path to your dataset location. The notebook is preserved for review, but is not yet a fully reproducible, leakage-free training pipeline. See the limitations below.
Important review notes / next steps
1. Split into training, validation and test before fitting imputers, encoders, scalers or data-dependent clipping. Current notebook fits preprocessing before splitting, causing leakage.
2. Replace per-column LabelEncoder with a fitted, saved encoder/ColumnTransformer pipeline, with unknown-category handling.
3. Tune the business-cost threshold on validation predictions, then evaluate once on untouched test data. The notebook currently tunes on test predictions.
4. ROC-AUC and average precision must use probabilities, not thresholded labels. Use src/evaluation.py.
5. Verify the notebook's stated 0.78 ROC-AUC, 0.427 KS and 0.59 threshold by rerunning corrected experiments; they are not independently verified here.
6. Fix X_train[train_idx] indexing for DataFrames (.iloc) or use NumPy arrays consistently; review SHAP output shape and feature-name alignment.
7. Save the full fitted preprocessing + model pipeline and feature schema before building a Streamlit app.
8. The cost weights FN=5 and FP=1 are assumptions, not validated real-world bank costs.
Deployment
Deployment is optional and not required to push this repository. A Streamlit demo should only be added after preprocessing and inference have been tested end-to-end.
Disclaimer
For educational/portfolio purposes only. Do not use for real credit decisions without robust validation, bias/fairness review, privacy safeguards and applicable compliance review.