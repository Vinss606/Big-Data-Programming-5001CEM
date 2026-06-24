# Financial Fraud Detection

A machine learning project that detects fraudulent financial transactions using the PaySim dataset.

## Project Structure

```
├── notebooks/          # Jupyter notebook with full analysis
│   └── fraud_detection.ipynb
├── dashboard/          # Streamlit interactive dashboard
│   └── app.py
├── outputs/
│   ├── graphs/         # EDA and model evaluation charts
│   │   ├── eda_overview.png
│   │   ├── model_comparison.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   └── model_results.csv
├── models/             # Trained model files
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── requirements.txt    # Python dependencies
└── README.md
```

## Dataset

**PaySim** – a synthetic mobile money transaction dataset with ~6.3 million rows.  
Download from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place `paysim.csv` in the project root (`Coding/`) before running the notebook.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **macOS/Linux:** `source .venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `notebooks/fraud_detection.ipynb` in VS Code (or Jupyter Lab) and run all cells top-to-bottom.

The notebook will:

- Load and explore the dataset (EDA)
- Engineer features and balance classes
- Train Logistic Regression, Random Forest, and XGBoost models
- Save evaluation charts to `outputs/graphs/`
- Save trained models to `models/`
- Write the Streamlit dashboard to `dashboard/app.py`

## Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Model Results

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 94.9%    | 33.8%     | 94.8%  | 49.8%    |
| Random Forest       | 99.99%   | 99.6%     | 99.9%  | 99.7%    |
| XGBoost             | 99.99%   | 99.6%     | 99.9%  | 99.8%    |

Random Forest and XGBoost both achieve near-perfect detection. The engineered features `balanceDiffOrig` and `amountToOldBalanceRatio` were the most predictive.

## Requirements

- Python 3.11+
- See `requirements.txt` for full package list
