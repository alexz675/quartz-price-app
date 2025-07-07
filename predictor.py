import streamlit as st
import pandas as pd
import joblib
import os
from datetime import date
import re
import numpy as np

@st.cache_data
def get_start_date():
    df = pd.read_excel("fake_parsed_data.xlsx", usecols=["日期"])
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    return df["日期"].min().date()

start_date = get_start_date()

# Load Best Model
log_file = "model_training_log.csv"

def load_best_model():
    fallback_model_path = "model_final.pkl"

    try:
        if not os.path.exists(log_file):
            raise FileNotFoundError("No model log found.")

        log_df = pd.read_csv(log_file)

        # Sort by R² score (descending), pick the top
        best_row = log_df.sort_values("r2", ascending=False).iloc[0]
        model_file = best_row["model_file"]

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file '{model_file}' not found.")

        return joblib.load(model_file), best_row

    except Exception as e:
        print(f"⚠️ {e}\n➡️ Loading fallback model: {fallback_model_path}")
        if not os.path.exists(fallback_model_path):
            raise FileNotFoundError(f"Fallback model '{fallback_model_path}' not found either.")
        
        fallback_model = joblib.load(fallback_model_path)
        return fallback_model, {"model_file": fallback_model_path, "note": "Fallback model loaded"}

# Example usage
model, model_info = load_best_model()

def excel_code_to_int(code):
    if isinstance(code, int):
        return code

    code = str(code).strip().upper()
    match = re.match(r"([A-Z]+)(\d+)$", code)
    if not match:
        if code.isdigit():
            return int(code)
        else:
            raise ValueError(f"Invalid code format: {code}")

    letters, digits = match.groups()
    letter_val = 0
    for char in letters:
        letter_val = letter_val * 26 + (ord(char) - ord('A'))
    return 100 + letter_val * 10 + int(digits)

def prepare_input(
    quartz_type=None,
    company_code=None,
    material_code=None,
    thickness=None,
    days_since_start=None,
    diameter=None,
    length=None,
    feature_order=None
):

    input_dict = {
        "quartz_type": int(quartz_type) if quartz_type is not None else 1,
        "company_code": int(company_code) if company_code is not None else 0,
        "material_code": int(material_code) if material_code is not None else 0,
        "thickness": float(thickness) if thickness is not None else 0.0,
        "days_since_start": int(days_since_start) if days_since_start is not None else 0,
        "diameter": float(diameter) if diameter is not None else 0.0,
        "length": float(length) if length is not None else 0.0,
    }

    input_df = pd.DataFrame([input_dict])

    if feature_order:
        for col in feature_order:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_order]

    return input_df

# === Load mapping files ===
mapping_file = "cleaned_mapping_file.xlsx"
quartz_df = pd.read_excel(mapping_file, sheet_name="quartz_map").dropna(how='all')
company_df = pd.read_excel(mapping_file, sheet_name="company_map").dropna(how='all')
material_df = pd.read_excel(mapping_file, sheet_name="material_map").dropna(how='all')

quartz_map = dict(zip(quartz_df["Quartz Type Name"].astype(str).str.strip(), quartz_df["Quartz Type Code"]))
company_map = dict(zip(company_df["Company Name"].astype(str).str.strip(), company_df["Company Code"]))
material_map = dict(zip(material_df["Material Name"].astype(str).str.strip(), material_df["Material Code"]))

# === Streamlit UI ===
st.title("Quartz Unit Price Predictor")
st.markdown("Enter the product details below to get a predicted unit price.")

quartz_input = st.selectbox("🔧 Quartz Type", options=list(quartz_map.keys()), placeholder="Type or select...")
company_input = st.selectbox("🏢 Company Name", options=list(company_map.keys()), placeholder="Type or select...")
material_input = st.selectbox("🧪 Material Name", options=list(material_map.keys()), placeholder="Type or select...")

length = st.number_input("📏 Length (mm)", min_value=0.0, step=1.0)
diameter = st.number_input("🔘 Diameter (mm)", min_value=0.0, step=1.0)
thickness = st.number_input(" Thickness (mm)", min_value=0.0, step=0.1, format="%.2f")
input_date = st.date_input("📅 Order Date", value=date.today(), format="YYYY-MM-DD")
days_since_start = (input_date - start_date).days

if st.button("💲 Predict Unit Price"):
    try:
        quartz_code = quartz_map.get(quartz_input, None)
        company_raw = company_map.get(company_input, None)
        material_code = material_map.get(material_input, None)

        try:
            company_code = excel_code_to_int(company_raw)
        except Exception as e:
            st.error(f"❌ Company code conversion error: {e}")
            st.stop()

        values = [quartz_code, company_code, material_code]
        if any([v is None or (isinstance(v, float) and pd.isna(v)) for v in values]):
            st.error("❌ Could not recognize some fields. Please check your input.")
            st.stop()

        input_df = prepare_input(
            quartz_type=quartz_code,
            company_code=company_code,
            material_code=material_code,
            thickness=thickness,
            days_since_start=days_since_start,
            diameter=diameter,
            length=length,
            feature_order=list(model.feature_names_in_)
        )

        raw_prediction = model.predict(input_df)[0]
        prediction = float(np.exp(raw_prediction))
        st.success(f"💰 Predicted Unit Price: ¥{prediction:.2f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


