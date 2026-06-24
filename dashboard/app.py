import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Mobile Money Fraud Detection Dashboard")

MODEL_PATH   = "models/random_forest_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"
RESULTS_PATH = "outputs/model_results.csv"
GRAPH_PATH   = "outputs/graphs"

model           = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

st.header("1. Model Performance")
if os.path.exists(RESULTS_PATH):
    st.dataframe(pd.read_csv(RESULTS_PATH))
else:
    st.warning("Run the notebook first to generate model_results.csv")

st.header("2. EDA & Results Graphs")
for graph in ["eda_overview.png", "model_comparison.png",
              "confusion_matrix.png", "feature_importance.png"]:
    path = os.path.join(GRAPH_PATH, graph)
    if os.path.exists(path):
        st.image(path, caption=graph.replace("_", " ").replace(".png", "").title())

st.header("3. Predict a Transaction")
col1, col2 = st.columns(2)
with col1:
    step             = st.number_input("Step",                    min_value=1,   value=1)
    transaction_type = st.selectbox("Transaction Type",
                                    ["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"])
    amount           = st.number_input("Amount",                  min_value=0.0, value=1000.0)
    oldbalanceOrg    = st.number_input("Sender Old Balance",      min_value=0.0, value=5000.0)
    newbalanceOrig   = st.number_input("Sender New Balance",      min_value=0.0, value=4000.0)
with col2:
    oldbalanceDest   = st.number_input("Receiver Old Balance",    min_value=0.0, value=1000.0)
    newbalanceDest   = st.number_input("Receiver New Balance",    min_value=0.0, value=2000.0)

input_data = pd.DataFrame([{
    "step": step, "amount": amount,
    "oldbalanceOrg": oldbalanceOrg, "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest, "newbalanceDest": newbalanceDest,
    "balanceDiffOrig":         oldbalanceOrg - newbalanceOrig,
    "balanceDiffDest":         newbalanceDest - oldbalanceDest,
    "amountToOldBalanceRatio": amount / (oldbalanceOrg + 1),
    "type_CASH_OUT": int(transaction_type == "CASH_OUT"),
    "type_DEBIT":    int(transaction_type == "DEBIT"),
    "type_PAYMENT":  int(transaction_type == "PAYMENT"),
    "type_TRANSFER": int(transaction_type == "TRANSFER"),
}])
for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[feature_columns]

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    if pred == 1:
        st.error("Prediction: FRAUDULENT Transaction")
    else:
        st.success("Prediction: Legitimate Transaction")

st.caption("Academic prototype â€” not connected to a real banking system.")
