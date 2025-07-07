# Quartz Price Prediction Machine Learning App (Sanitized)

This repository contains a **Streamlit-based machine learning app** for predicting the unit price of quartz materials using historical inventory and specification data.

**Important: This repository uses sanitized code and fully fake/generated data. No confidential company information is included so you can't train a model, but you can still see how the logic works.**

---

## Features

- 📊 Interactive prediction app with dropdowns for product attributes, makes prediction on unit price after everything is entered
- 🧠 Model trainer using `HistGradientBoostingRegressor`, intreactive training app that helps employees easily train new models with new data
- 🔁 Undo functionality for parsed data uploads
- 🗂️ Cleaned and anonymized mapping system for quartz types, companies, and materials

---

## File Overview

| File | Description |
|------|-------------|
| `predictor.py` | The **Streamlit app** for users to predict the unit price of quartz using a trained model. Requires the fake parsed data and sanitized mapping file. |
| `modeltrain.py` | A **Streamlit app** for training the machine learning model. Parses uploaded inventory data, engineers features, trains the model, and logs performance. |
| `model_final.pkl` | A **trained model** file built using fake data. This model allows the app to work out of the box without requiring training. |
| `cleaned_mapping_file.xlsx` | Contains **anonymized mapping tables** for: <br/>• Quartz types → `Quartz Type Name` to `Quartz Type Code`<br/>• Company names → `Company X` to codes<br/>• Material names → `Material X` to codes |
| `fake_parsed_data.xlsx` | Fake but **structure-preserving parsed inventory data**. This mimics the real format but is completely randomized. This stores the data that the parsing function creates when training the model, and is the data new models are trained off of (Accumulation of past and current data)|
| `fake_parsed_data_backup.xlsx` | Backup copy of the above, used by the app's undo function. |
| `model_training_log.csv` | Keeps track of trained models, including RMSE and R² performance. Uses fake training history. |

---

## Data Anonymization & Security

This repo follows strong sanitization practices:

- ✅ Company, material, and quartz type names are **fully replaced** with labels like `Company 1`, `Material 1`, etc.
- ✅ All values in Excel files (`parsed_data`, `backup`, etc.) are **randomized** but **parsable and realistic**.
- ✅ Serial numbers are retained in format for parsing, but randomized in content.
- ✅ No internal file paths or private business logic is exposed.

---

## Disclaimer

> This project is for demonstration and educational purposes only.  
> **Do not use this repository for actual commercial operations or pricing predictions.**

---

## How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run either app:
    - Model Trainer:
      ```bash
      streamlit run modeltrain.py
      ```
    - Price Predictor:
      ```bash
      streamlit run predictor.py
      ```

---

## Contact

For questions, feel free to reach out through GitHub or include this project on your resume/portfolio!


