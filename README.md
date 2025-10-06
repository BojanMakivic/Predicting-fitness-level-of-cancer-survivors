# Predicting Maximal Work Capacity of Cancer Survivors

A small project that uses machine learning models to predict the maximal work capacity (Watt) achieved during a maximal cycle ergometry test for cancer survivors. The goal is to support clinical assessments by estimating a patient's peak exercise workload from basic demographic and anthropometric features.

## Table of Contents
- Project overview
- Dataset
- Models used
- Evaluation
- How to run / Reproduce
- Requirements
- Results (summary)
- Notes, limitations & next steps
- Contributing
- License & contact

## Project overview
After cancer treatment (surgery, chemotherapy, radiotherapy), many patients undergo physical rehabilitation to restore fitness (endurance, strength, flexibility). The cycle ergometer maximal test is a standard tool for assessing maximal aerobic/work capacity but requires specialized equipment and supervision. This project aims to predict the highest workload (Watt) achieved by a patient using easily available data (age, sex, weight, height, etc.) to support triage and planning of rehabilitation.

Recommended reading: Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow (if you want a practical ML reference).

## Dataset
- Features include: age, gender, body weight, body height, and maximal workload (Watt) measured on a cycle ergometer.
- The dataset used in this repository is located in the repository (check the `data/` directory or the notebooks for the exact filename).
- Data preprocessing steps used in the notebooks:
  - Missing value handling
  - Encoding categorical variables (e.g., gender)
  - Feature scaling where needed
  - Train/test split and cross-validation

## Models
Three supervised regression models were trained and compared:
- Linear Regression
- Support Vector Regressor (SVR)
- Random Forest Regressor

Each model was trained to predict maximal workload (Watt). The notebooks include hyperparameter tuning and cross-validation for fair comparisons.

## Evaluation
Common regression metrics used to evaluate models:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE) / Root Mean Squared Error (RMSE)
- R-squared (R²)

Use cross-validation scores to compare model robustness and a hold-out test set for final evaluation.

## How to run / Reproduce
1. Clone the repository:
   git clone https://github.com/BojanMakivic/Predicting-fitness-level-of-cancer-survivors.git
2. Create and activate a virtual environment (recommended):
   python -m venv .venv
   source .venv/bin/activate  # Linux / macOS
   .venv\Scripts\activate     # Windows
3. Install dependencies:
   pip install -r requirements.txt
   (If no requirements file is available, install common packages:
    pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab)
4. Launch Jupyter and open the notebooks:
   jupyter lab
5. Follow the notebooks in order:
   - Data exploration & preprocessing
   - Model training & hyperparameter tuning
   - Evaluation & comparison

If there are scripts for training/evaluation (e.g., train.py), run them with:
   python train.py --config configs/your_config.yml

## Requirements
- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyterlab / notebook

Add any other packages used in the notebooks (e.g., joblib, xgboost) to `requirements.txt`.

## Results (summary)
- A concise summary of model performance and which model worked best should be added here after running the notebooks. Example format:
  - Linear Regression: MAE = XX.X, RMSE = XX.X, R² = 0.XX
  - SVR: MAE = XX.X, RMSE = XX.X, R² = 0.XX
  - Random Forest: MAE = XX.X, RMSE = XX.X, R² = 0.XX
- Include plots such as predicted vs actual, residuals, and feature importances.

## Notes, limitations & next steps
- Sample size and representativeness: Ensure the dataset is representative of the patient population where the model will be used.
- Clinical validation: Model predictions should be validated prospectively before any clinical deployment.
- Potential extensions:
  - Add more clinical predictors (e.g., comorbidities, treatment type, time since treatment)
  - Calibrate models for different subgroups (age ranges, sex)
  - Use ensembles or gradient boosting (e.g., XGBoost, LightGBM)
  - Deploy as a simple web app for clinician use (Flask / Streamlit)

## Contributing
Contributions are welcome. Please:
1. Fork the repo
2. Create a feature branch
3. Open a pull request describing your changes

## License & Contact
- MIT license 
- Author: Bojan Makivić
- Contact / Questions: https://github.com/BojanMakivic
