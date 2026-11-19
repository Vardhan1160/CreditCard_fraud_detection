import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ── Load Model & Scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('model/model.pkl')  if os.path.exists('model/model.pkl')  else None
    scaler = joblib.load('model/scaler.pkl') if os.path.exists('model/scaler.pkl') else None
    return model, scaler

model, scaler = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("💳 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Single Prediction", "📂 Batch Prediction"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.info(
    "Uses the real [Kaggle Credit Card Fraud dataset]"
    "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — "
    "284,807 transactions, 492 fraud (0.17%)."
)
st.sidebar.markdown("**About**")
st.sidebar.info(
    "Detects fraudulent credit card transactions using XGBoost + SMOTE. "
    "Built as part of my final year B.Tech project."
)

# ──────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("💳 Credit Card Fraud Detection")
    st.markdown("### Real-World Kaggle Dataset · XGBoost + SMOTE")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Algorithm",        "XGBoost (best)")
    col2.metric("Dataset Size",     "284,807 transactions")
    col3.metric("Fraud Cases",      "492  (0.17%)")
    col4.metric("Imbalance Fix",    "SMOTE")

    st.markdown("---")
    st.markdown("""
    ## About This Project

    Credit card fraud causes billions of dollars in losses every year.
    This project uses the **real Kaggle Credit Card Fraud dataset** (from ULB) to build
    a model that automatically identifies whether a transaction is **fraudulent** or **legitimate**.

    ### Dataset Features
    | Feature | Description |
    |---|---|
    | **V1 – V28** | PCA-transformed anonymized features |
    | **Time** | Seconds elapsed since the first transaction |
    | **Amount** | Transaction amount (scaled) |
    | **Class** | 0 = Normal · 1 = Fraud |

    ### Why is this hard?
    - Only **0.17%** of transactions are fraud → extreme class imbalance.
    - Plain accuracy is misleading (a model that always says "Normal" gets 99.83%!).
    - We use **SMOTE** (Synthetic Minority Oversampling) on training data +
      **F1 Score** and **ROC-AUC** as evaluation metrics.

    ### Models Trained
    1. Logistic Regression (baseline)
    2. Random Forest
    3. **XGBoost** ✅ (best — saved to `model/model.pkl`)

    ### Tech Stack
    Python · Scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Streamlit · Plotly
    """)

    if model is None:
        st.warning("⚠️ Model not found. Run `python download_data.py` then `python train.py` first.")
    else:
        st.success("✅ Model loaded successfully! Use the sidebar to make predictions.")

# ──────────────────────────────────────────────────────────────────────────────
# SINGLE PREDICTION PAGE
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Single Prediction":
    st.title("🔍 Single Transaction Prediction")
    st.markdown("Enter transaction details to check if it's fraudulent.")
    st.markdown("---")

    if model is None:
        st.error("❌ Model not found. Run `python train.py` first.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Info")
        time_val   = st.number_input("Time (seconds since first transaction)", value=0.0, step=1.0)
        amount_val = st.number_input("Amount (₹ or $)", value=50.0, step=0.5, min_value=0.0)

        st.subheader("PCA Features (V1 – V28)")
        st.caption("Anonymized PCA features from the original card data.")

        v_vals = {}
        for i in range(1, 29):
            v_vals[f'V{i}'] = st.number_input(f"V{i}", value=0.0, step=0.01, key=f"v{i}")

    with col2:
        st.subheader("Prediction Result")
        st.markdown(" ")

        if st.button("🚀 Predict", use_container_width=True):
            # Build input dataframe
            input_dict = {'Time': [time_val], 'Amount': [amount_val]}
            input_dict.update({f'V{i}': [v_vals[f'V{i}']] for i in range(1, 29)})
            input_df = pd.DataFrame(input_dict)

            # Scale Time and Amount using saved scaler
            if scaler is not None:
                input_df['Amount'] = scaler.transform(input_df[['Amount']])
                input_df['Time']   = scaler.transform(input_df[['Time']])
            else:
                # Fallback rough normalization if scaler not saved
                input_df['Time']   = (time_val - 94813) / 47488
                input_df['Amount'] = (amount_val - 88) / 250

            # Reorder columns to match training
            feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            input_df = input_df[feature_cols]

            prediction  = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.error("🚨 **FRAUD DETECTED!**")
            else:
                st.success("✅ **Normal Transaction**")

            st.metric("Fraud Probability", f"{probability * 100:.2f}%")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = probability * 100,
                title = {'text': "Fraud Risk Score (%)"},
                gauge = {
                    'axis'     : {'range': [0, 100]},
                    'bar'      : {'color': "crimson" if probability > 0.5 else "green"},
                    'steps'    : [
                        {'range': [0, 30],  'color': '#d4edda'},
                        {'range': [30, 60], 'color': '#fff3cd'},
                        {'range': [60, 100],'color': '#f8d7da'},
                    ],
                    'threshold': {
                        'line'     : {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value'    : 50
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# BATCH PREDICTION PAGE
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown("Upload a CSV to run predictions on multiple transactions.")
    st.markdown("---")

    if model is None:
        st.error("❌ Model not found. Run `python train.py` first.")
        st.stop()

    st.info("📌 CSV must have columns: **Time, V1–V28, Amount** (Class is optional for evaluation)")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} transactions")
        st.dataframe(df.head(10))

        required_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            st.error(f"❌ Missing columns: {missing}")
        else:
            if st.button("🚀 Run Predictions", use_container_width=True):
                with st.spinner("Running predictions..."):
                    X = df[required_cols].copy()

                    # Scale using saved scaler or StandardScaler fallback
                    if scaler is not None:
                        X['Amount'] = scaler.transform(X[['Amount']])
                        X['Time']   = scaler.transform(X[['Time']])
                    else:
                        X['Amount'] = StandardScaler().fit_transform(X[['Amount']])
                        X['Time']   = StandardScaler().fit_transform(X[['Time']])

                    preds = model.predict(X)
                    probs = model.predict_proba(X)[:, 1]

                    df['Prediction']        = preds
                    df['Fraud_Probability'] = probs.round(4)
                    df['Status']            = df['Prediction'].map({0: '✅ Normal', 1: '🚨 Fraud'})

                st.markdown("### Results")
                st.dataframe(df[['Time', 'Amount', 'Status', 'Fraud_Probability']])

                # Summary metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Transactions", f"{len(df):,}")
                c2.metric("Fraud Detected",     int(df['Prediction'].sum()))
                c3.metric("Fraud Rate",         f"{df['Prediction'].mean()*100:.2f}%")

                # Fraud probability distribution chart
                fig = px.histogram(
                    df, x='Fraud_Probability',
                    color='Status',
                    nbins=50,
                    title="Fraud Probability Distribution",
                    labels={'Fraud_Probability': 'Fraud Probability'},
                    color_discrete_map={'✅ Normal': 'steelblue', '🚨 Fraud': 'crimson'}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Confusion matrix — only if actual Class labels are available
                if 'Class' in df.columns:
                    st.markdown("### Confusion Matrix (vs Actual Labels)")
                    cm = confusion_matrix(df['Class'], df['Prediction'])
                    fig2 = px.imshow(
                        cm,
                        text_auto=True,
                        x=['Predicted Normal', 'Predicted Fraud'],
                        y=['Actual Normal', 'Actual Fraud'],
                        color_continuous_scale='Blues',
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    roc = roc_auc_score(df['Class'], probs)
                    f1  = f1_score(df['Class'], preds)
                    m1, m2 = st.columns(2)
                    m1.metric("ROC-AUC Score", f"{roc:.4f}")
                    m2.metric("F1 Score",      f"{f1:.4f}")

                # Download results
                csv_out = df.to_csv(index=False)
                st.download_button(
                    label    = "⬇️ Download Results as CSV",
                    data     = csv_out,
                    file_name= "predictions.csv",
                    mime     = "text/csv"
                )
