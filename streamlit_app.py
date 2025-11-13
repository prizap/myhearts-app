# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from io import BytesIO
from typing import Optional

# CONFIG: path to model file inside the repo. Replace with your actual filename if different.
# e.g. saved_models/all_models/model_RandomForest_20251113_152301.pkl
MODEL_DIR = os.getenv("MODEL_DIR", "saved_models/all_models")
# If you know the exact filename, you can set MODEL_FILE to that name; otherwise leave None to auto-select latest .pkl
MODEL_FILE = os.getenv("MODEL_FILE", "model_SVC_20251113_082336.pkl")  # empty means auto-find latest

FEATURE_COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]


@st.cache_resource(show_spinner=False)
def load_model(model_dir: str = MODEL_DIR, model_file: Optional[str] = MODEL_FILE):
    """Load a joblib model from the given directory. If model_file is empty, load the latest .pkl file found.
    Returns: (model_filename, loaded_pipeline)
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # list .pkl files
    files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not files:
        raise FileNotFoundError(f"No .pkl model files found in {model_dir}")

    if model_file:
        if model_file not in files and not os.path.exists(os.path.join(model_dir, model_file)):
            raise FileNotFoundError(f"Requested model file not found: {model_file}")
        chosen = model_file
    else:
        files_sorted = sorted(files)
        chosen = files_sorted[-1]  # pick the latest by name

    path = os.path.join(model_dir, chosen)
    pipe = joblib.load(path)
    return chosen, pipe


st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="centered")
st.title("Demo Prediksi Penyakit Jantung")

st.markdown("This app loads a saved scikit-learn pipeline (preprocessor + model) from `saved_models/all_models/` andperforms single or batch predictions.")

# Try to load model
try:
    model_name, model_pipe = load_model()
    st.success(f"Loaded model: {model_name}")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# Sidebar: single input
st.sidebar.header("Single prediction input")
age = st.sidebar.number_input("age", value=63, min_value=1, max_value=120)
sex = st.sidebar.selectbox("sex", options=[0,1], index=1)
cp = st.sidebar.selectbox("cp (chest pain type)", options=[0,1,2,3], index=3)
trestbps = st.sidebar.number_input("trestbps", value=145)
chol = st.sidebar.number_input("chol", value=233)
fbs = st.sidebar.selectbox("fbs", options=[0,1], index=0)
restecg = st.sidebar.selectbox("restecg", options=[0,1,2], index=0)
thalach = st.sidebar.number_input("thalach", value=150)
exang = st.sidebar.selectbox("exang", options=[0,1], index=0)
oldpeak = st.sidebar.number_input("oldpeak", value=2.3, format="%.2f")
slope = st.sidebar.selectbox("slope", options=[0,1,2], index=0)
ca = st.sidebar.number_input("ca", value=0, min_value=0, max_value=5)
thal = st.sidebar.number_input("thal", value=1, min_value=0, max_value=5)

if st.sidebar.button("Predict (single)"):
    row = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    try:
        pred = model_pipe.predict(row).tolist()
        proba = model_pipe.predict_proba(row)[:,1].tolist() if hasattr(model_pipe, "predict_proba") else None
        st.write("**Prediction (label):**", pred[0])
        if proba is not None:
            st.write("**Probability (class=1):**", round(proba[0], 4))
    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.markdown("---")

# Batch prediction
st.header("Batch prediction (upload CSV)")
uploaded = st.file_uploader("Upload CSV with columns: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())
    required = FEATURE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns in uploaded CSV: {missing}")
    else:
        if st.button("Run batch prediction"):
            # gunakan DataFrame slice sehingga preprocessor bisa memilih kolom berdasarkan nama
            df_input = df[required].copy()
            try:
                preds = model_pipe.predict(df_input)
                try:
                    probs = model_pipe.predict_proba(df_input)[:,1]
                except Exception:
                    probs = None
                df_out = df.copy()
                df_out['prediction'] = preds
                if probs is not None:
                    df_out['probability'] = probs
                st.write(df_out.head())

                # provide download
                towrite = BytesIO()
                df_out.to_csv(towrite, index=False)
                towrite.seek(0)
                st.download_button("Download predictions CSV", data=towrite, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

st.markdown("---")
st.write("Notes:\n- Ensure uploaded CSV has the exact columns and data types used at training.\n- For large models, consider storing model externally (S3/GCS) and loading in startup.")