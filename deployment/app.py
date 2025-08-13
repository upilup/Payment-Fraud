import streamlit as st
import importlib

st.set_page_config(page_title="Payment Fraud Detection", layout="wide", initial_sidebar_state="expanded")

# dukung run dari root ataupun dari folder deployment
try:
    home = importlib.import_module("deployment.home")
    eda  = importlib.import_module("deployment.eda")
    pred = importlib.import_module("deployment.prediction")
except Exception:
    import home, eda
    import prediction as pred

with st.sidebar:
    st.write("# Navigation")
    page = st.radio("Page", ["Home", "EDA", "Predict Fraud"])

if page == "Home":
    home.home()
elif page == "EDA":
    eda.eda()
else:
    pred.run()