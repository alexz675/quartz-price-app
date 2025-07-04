import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore
from datetime import datetime
import os
import re
import csv

def parse_serial(serial):
    result = {}
    if not isinstance(serial, str) or len(serial) < 7:
        return None
    if not serial.startswith('0'):
        return None  # Not quartz
    region_digit = serial[1]
    if region_digit != '2':
        return None  # ❌ Skip non-mainland
    result["存货编码"] = serial
    result["is_quartz"] = True
    result["region"] = 2  # ✅ Only mainland
    result["quartz_type"] = int(serial[2]) if serial[2].isdigit() else None

    code = serial[3:5]
    if code.isdigit():
        result["company_code"] = int(code)
    elif re.match(r'[A-Z][0-9]', code):
        result["company_code"] = 100 + (ord(code[0]) - ord('A')) * 10 + int(code[1])
    else:
        result["company_code"] = None

    mat_code = serial[5:7]
    result["material_code"] = int(mat_code) if mat_code.isdigit() else None
    return result

def parse_dimensions(spec):
    if not isinstance(spec, str):
        return {
            "length": 0, "width": 0, "thickness": 0,
            "diameter": 0, "is_round": 0, "has_dimensions": 0
        }

    spec = spec.replace("×", "x").replace("X", "x").replace("*", "x").replace("Φ", "φ")
    dims = {
        "length": 0, "width": 0, "thickness": 0,
        "diameter": 0, "is_round": 0, "has_dimensions": 1
    }

    match1 = re.match(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", spec)
    match2 = re.match(r"φ(\d+(?:\.\d+)?)/\d+(?:\.\d+)?x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", spec)
    match3 = re.match(r"φ(\d+(?:\.\d+)?)/\d+(?:\.\d+)?x(\d+(?:\.\d+)?)", spec)
    match4 = re.match(r"φ(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", spec)
    match5 = re.match(r"φ(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", spec)
    has_plus_minus = "+" in spec or "-" in spec

    try:
        if match1:
            dims["width"] = float(match1.group(1))
            dims["thickness"] = float(match1.group(2))
            dims["length"] = float(match1.group(3))
        elif match2:
            dims["diameter"] = float(match2.group(1))
            dims["thickness"] = float(match2.group(2))
            dims["length"] = float(match2.group(3))
            dims["is_round"] = 1
        elif match3:
            dims["diameter"] = float(match3.group(1))
            dims["thickness"] = float(match3.group(2))
            dims["is_round"] = 1
        elif match4:
            dims["diameter"] = float(match4.group(1))
            dims["thickness"] = float(match4.group(2))
            dims["length"] = float(match4.group(3))
            dims["is_round"] = 1
        elif match5:
            dims["diameter"] = float(match5.group(1))
            dims["length"] = float(match5.group(2))
            dims["is_round"] = 1
        elif has_plus_minus:
            dims["has_dimensions"] = 0
        else:
            dims["has_dimensions"] = 0
    except:
        dims["has_dimensions"] = 0

    return dims


# === UI ===
st.title("📦 Mainland Quartz Price Model Trainer")

uploaded_file = st.file_uploader("Upload Excel File", type=["xls", "xlsx"])

if "parsed_successfully" not in st.session_state:
    st.session_state.parsed_successfully = False

if "parsed_df" not in st.session_state:
    st.session_state.parsed_df = None

if uploaded_file:
    st.info("📋 File uploaded. Click below to parse and confirm before saving.")

    if st.button("📄 Parse Uploaded Data"):
        try:
            df = pd.read_excel(uploaded_file, engine="xlrd" if uploaded_file.name.endswith(".xls") else None)
            if not all(col in df.columns for col in ["存货编码", "日期", "规格型号", "原币单价"]):
                st.error("❌ Missing required columns.")
            else:
                parsed_data = []
                for _, row in df.iterrows():
                    serial_info = parse_serial(str(row["存货编码"]))
                    if serial_info:
                        dims = parse_dimensions(row["规格型号"])
                        weight = 0
                        if isinstance(row.get("主计量"), str) and "kg" in row["主计量"].lower():
                            try:
                                weight = float(row["数量"])
                            except:
                                weight = 0
                        entry = {
                            **serial_info,
                            "日期": pd.to_datetime(row["日期"], errors="coerce").date(),
                            "规格型号": row["规格型号"],
                            "原币单价": row["原币单价"],
                            "weight": weight,
                            **dims
                        }
                        parsed_data.append(entry)

                new_df = pd.DataFrame(parsed_data)
                new_df = new_df.dropna(subset=["region", "quartz_type", "company_code", "material_code"])

                st.session_state.parsed_df = new_df  # ✅ Save parsed data for next interaction
                st.success("✅ Parsed. Click confirm to save.")

        except Exception as e:
            st.error(f"❌ Failed to parse: {e}")

# === Confirm and Save Button: Only shown if there's parsed data
if st.session_state.parsed_df is not None:
    if st.button("✅ Confirm and Save Parsed Data"):
        try:
            existing = pd.read_excel("fake_parsed_data.xlsx") if os.path.exists("fake_parsed_data.xlsx") else pd.DataFrame()

            # 🔁 Backup current data
            backup_path = "fake_parsed_data_backup.xlsx"
            if not existing.empty:
                existing.to_excel(backup_path, index=False)

            combined = pd.concat([existing, st.session_state.parsed_df], ignore_index=True)
            combined.to_excel("fake_parsed_data.xlsx", index=False)
            st.session_state.parsed_successfully = True
            st.success(f"✅ Parsed successfully, {len(st.session_state.parsed_df)} rows added. Total data number = {len(combined)}")
            st.session_state.parsed_df = None  # ✅ Clear after save

        except Exception as e:
            st.error(f"❌ Failed to save parsed data: {e}")

# === Undo Button
if os.path.exists("fake_parsed_data_backup.xlsx"):
    if st.button("↩️ Undo Last Upload"):
        try:
            # Read backup
            backup_df = pd.read_excel("fake_parsed_data_backup.xlsx")

            # Restore backup as current
            backup_df.to_excel("fake_parsed_data.xlsx", index=False)

            # ✅ Create a fresh backup to preserve current state post-undo
            backup_df.to_excel("fake_parsed_data_backup.xlsx", index=False)

            st.success(f"↩️ Reverted to previous state. Current data has {len(backup_df)} rows.")
        except Exception as e:
            st.error(f"❌ Undo failed: {e}")




train_model = st.button("🧠 Train Model")  # <-- always define it
# === Train Model on Button Click ===
if train_model:
    if not st.session_state.parsed_successfully:
        st.error("❌ No parsed inventory file found. Upload and parse data first.")
    else:
        df = pd.read_excel("fake_parsed_data.xlsx")
        df = df[(df["原币单价"] > 0) & (df["原币单价"] < 3000)]
        df["原币单价"] = df["原币单价"].astype(float)
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.sort_values("日期")
        df["days_since_start"] = (df["日期"] - df["日期"].min()).dt.days

        Q1, Q3 = df["原币单价"].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df = df[(df["原币单价"] >= Q1 - 1.5 * IQR) & (df["原币单价"] <= Q3 + 1.5 * IQR)]

        if len(df) > 30:
            df["zscore"] = zscore(df["原币单价"])
            df = df[df["zscore"].abs() < 3]
            df.drop(columns=["zscore"], inplace=True)

        for col in ["material_code", "company_code", "quartz_type"]:
            df = df[df[col].isin(df[col].value_counts()[lambda x: x > 10].index)]

        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0)

        X = df[["quartz_type", "company_code", "material_code", "weight", "days_since_start", "diameter", "length"]]
        y = np.log1p(df["原币单价"])
        weights = np.linspace(1, 2, len(X))
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

        best_rmse, best_iter = float("inf"), 0
        for n in range(50, 401, 50):
            model = HistGradientBoostingRegressor(max_iter=n, random_state=42)
            model.fit(X_train, y_train, sample_weight=w_train)
            y_pred = np.expm1(model.predict(X_test))
            rmse = mean_squared_error(np.expm1(y_test), y_pred, squared=False)
            if rmse < best_rmse:
                best_rmse, best_iter = rmse, n

        

        final_model = HistGradientBoostingRegressor(max_iter=best_iter, random_state=42)
        final_model.fit(X_train, y_train, sample_weight=w_train)
        final_rmse = mean_squared_error(np.expm1(y_test), np.expm1(final_model.predict(X_test)), squared=False)
        final_r2 = r2_score(np.expm1(y_test), np.expm1(final_model.predict(X_test)))

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        versioned_filename = f"mainland_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        joblib.dump(final_model, versioned_filename)
        # Save to model_log.csv
        log_file = "model_training_log.csv"
        log_entry = {
            "model_file": versioned_filename,
            "trained_at": timestamp,
             "rmse": round(final_rmse, 2),
             "r2": round(final_r2, 4)
         }

        if os.path.exists(log_file):
            log_df = pd.read_csv(log_file)
            log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            log_df = pd.DataFrame([log_entry])

            log_df.to_csv(log_file, index=False)

            st.success(f"✅ Model trained! RMSE: {final_rmse:.2f}, R²: {final_r2:.4f}")


# Always show training log if available
st.header("📜 Model Training Log")

if os.path.exists("model_training_log.csv"):
    log_df = pd.read_csv("model_training_log.csv")
    log_df = log_df.sort_values("trained_at", ascending=False)
    st.dataframe(log_df, use_container_width=True)
else:
    st.info("No training log yet. Train your first model to get started.")
