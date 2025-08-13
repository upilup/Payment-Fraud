# deployment/prediction.py
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

# ==== Lokasi artefak (urutan pencarian) ====
ARTIFACT_CANDIDATES = [
    Path("deployment/model.pkl"),
    Path("artifacts/model_pipeline.joblib"),
    Path("xgboost_best_model.pkl"),
]
META_CANDIDATES = [
    Path("deployment/model_meta.json"),
    Path("artifacts/model_meta.json"),
    Path("model_meta.json"),
]
THRESHOLD_CANDIDATES = [
    Path("deployment/best_threshold.pkl"),
    Path("artifacts/best_threshold.pkl"),
    Path("best_threshold.pkl"),
]

# Fallback mapping dari notebook kamu
DEFAULT_CATEGORY_PROB = {"shopping": 0.344749, "electronics": 0.328588, "food": 0.329321}


def _find_first(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def load_artifacts() -> Tuple[Any, Dict[str, Any], float]:
    """Load pipeline, meta (opsional), dan threshold (opsional) dengan fallback aman."""
    # meta dulu (agar pesan error bisa kasih hint versi)
    meta: Dict[str, Any] = {}
    m = _find_first(META_CANDIDATES)
    if m:
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    model_path = _find_first(ARTIFACT_CANDIDATES)
    if not model_path:
        raise FileNotFoundError("Model artifact tidak ditemukan (deployment/model.pkl atau lokasi cadangan).")

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        vers = meta.get("versions", {})
        raise RuntimeError(
            f"Gagal load model ({model_path}). Cek kompatibilitas scikit-learn/xgboost.\n"
            f"Versi saat training (meta): {vers}\nDetail: {e}"
        )

    thr = 0.10
    t = _find_first(THRESHOLD_CANDIDATES)
    if t:
        try:
            thr = float(joblib.load(t))
        except Exception:
            pass
    if "threshold" in meta:
        try:
            thr = float(meta["threshold"])
        except Exception:
            pass

    return pipeline, meta, float(thr)


@st.cache_resource
def _load():
    """Cached wrapper supaya artefak tidak di-load berulang saat navigasi."""
    return load_artifacts()


def _align_categories_to_training(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Samakan kapitalisasi & kategori dengan OneHotEncoder di pipeline training."""
    pre = None
    for step in getattr(pipeline, "named_steps", {}).values():
        if isinstance(step, ColumnTransformer):
            pre = step
            break
    if pre is None:
        return df

    ohe, cat_cols = None, []
    for name, trans, cols in pre.transformers_:
        if isinstance(trans, OneHotEncoder):
            ohe, cat_cols = trans, list(cols)
            break
        if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
            maybe = trans.named_steps["onehot"]
            if isinstance(maybe, OneHotEncoder):
                ohe, cat_cols = maybe, list(cols)
                break
    if ohe is None:
        return df

    def _align_case(s: pd.Series, training_cats):
        lut = {str(c).casefold(): c for c in training_cats}
        return s.astype(str).map(lambda x: lut.get(x.casefold(), x))

    df2 = df.copy()
    for i, col in enumerate(cat_cols):
        if col not in df2.columns:
            continue
        df2[col] = _align_case(df2[col], ohe.categories_[i])
        train_cats = set(map(str, ohe.categories_[i]))
        unk = ~df2[col].astype(str).isin(train_cats)
        if unk.any():
            fallback = "other" if "other" in train_cats else list(train_cats)[0]
            df2.loc[unk, col] = fallback
    return df2


def ensure_features(raw_df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """Bentuk fitur rekayasa yang dipakai saat training (jika belum ada)."""
    df = raw_df.copy()

    if "isNight" not in df.columns and "hour" in df.columns:
        df["isNight"] = ((df["hour"] >= 21) | (df["hour"] < 6)).astype(int)
    if "isHighRiskPayment" not in df.columns and "paymentMethod" in df.columns:
        df["isHighRiskPayment"] = (df["paymentMethod"].astype(str).str.casefold() == "paypal").astype(int)
    if "time_bin" not in df.columns and "hour" in df.columns:
        df["time_bin"] = df["hour"].astype(int)

    cat_prob_map = meta["category_prob_map"] if isinstance(meta.get("category_prob_map"), dict) else DEFAULT_CATEGORY_PROB
    if "category_prob" not in df.columns and "Category" in df.columns:
        df["category_prob"] = df["Category"].map(cat_prob_map).fillna(np.mean(list(cat_prob_map.values())))
    if "category_deviation" not in df.columns:
        df["category_deviation"] = 1.0 - df["category_prob"]

    expected = [
        "numItems", "localTime", "paymentMethod", "Category", "isHighRiskPayment",
        "hour", "isNight", "risk_score", "time_bin", "transaction_velocity",
        "payment_age_ratio", "category_prob", "category_deviation", "temporal_risk_window",
    ]
    if all(c in df.columns for c in expected):
        df = df[expected]
    return df


def predict_df(df: pd.DataFrame, pipeline, meta: Dict[str, Any], threshold: float) -> pd.DataFrame:
    """Prediksi proba & label dengan threshold."""
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    df = ensure_features(df, meta)
    df = _align_categories_to_training(df, pipeline)

    proba = pipeline.predict_proba(df)[:, 1]
    pred = (proba >= float(threshold)).astype(int)

    out = df.copy()
    out["fraud_proba"] = proba
    out["fraud_pred"] = pred
    return out


# ==== Halaman Streamlit (dipanggil dari app.py) ====
def run():
    st.header("Prediksi Transaksi")
    pipeline, meta, threshold = _load()

    with st.form("fraud_form"):
        pm  = st.selectbox("Payment Method", ["creditcard", "storecredit", "paypal"], index=0)
        cat = st.selectbox("Category", ["shopping", "electronics", "food"], index=0)
        num = st.number_input("Number of Items", min_value=1, max_value=50, value=2, step=1)
        ltm = st.number_input("Local Time (≈4.70-5.05)", value=4.92, step=0.01, format="%.2f")
        hr  = st.number_input("hour (0-23)", min_value=0, max_value=23, value=14, step=1)
        rsk = st.number_input("Risk Score (0-1)", min_value=0.0, max_value=1.0, value=0.15, step=0.01, format="%.2f")
        vel = st.number_input("Transaction Velocity", min_value=0.0, value=3.0, step=0.1, format="%.2f")
        par = st.number_input("Payment Age Ratio (0-1)", min_value=0.0, max_value=1.0, value=0.80, step=0.01, format="%.2f")
        trw = st.selectbox("Temporal Risk Window", [0, 1], index=0)

        if st.form_submit_button("Predict"):
            df = pd.DataFrame([{
                "paymentMethod": pm, "Category": cat, "numItems": num,
                "localTime": ltm, "hour": hr, "risk_score": rsk,
                "transaction_velocity": vel, "payment_age_ratio": par,
                "temporal_risk_window": trw
            }])
            res = predict_df(df, pipeline, meta, threshold)
            prob = float(res["fraud_proba"].iloc[0]); pred = int(res["fraud_pred"].iloc[0])
            st.metric("Fraud Probability", f"{prob:.4f}")
            st.metric("Prediction", "Fraud" if pred == 1 else "Legitimate")

    # st.subheader("Batch CSV")
    # st.write("Kolom minimal: paymentMethod, Category, numItems, localTime, hour, risk_score, "
    #          "transaction_velocity, payment_age_ratio, temporal_risk_window. Fitur rekayasa dibentuk otomatis.")
    # up = st.file_uploader("Upload CSV", type=["csv"])
    # if up is not None:
    #     try:
    #         df = pd.read_csv(up)
    #         res = predict_df(df, pipeline, meta, threshold)
    #         st.dataframe(res.head(50), use_container_width=True)
    #         st.download_button("Download predictions.csv",
    #                            res.to_csv(index=False).encode("utf-8"),
    #                            "predictions.csv", "text/csv")
    #     except Exception as e:
    #         st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    # Memungkinkan jalankan langsung file ini untuk debug cepat
    st.set_page_config(page_title="Payment Fraud Detection", layout="wide")
    run()