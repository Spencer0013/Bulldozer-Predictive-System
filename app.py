import streamlit as st
import pandas as pd
import joblib
import os
import tempfile

from bullprediction.conponents.data_ingestion import DataIngestion
from bullprediction.entity import DataIngestionConfig, DataTransformationConfig
from bullprediction.utils.common import read_yaml
from bullprediction.constants import CONFIG_FILE_PATH

# ── Load config & initialize components ───────────────
config = read_yaml(CONFIG_FILE_PATH)
ingest_cfg = DataIngestionConfig(**config['data_ingestion'])
trans_cfg = DataTransformationConfig(**config['data_transformation'])

data_ingestion = DataIngestion(ingest_cfg)

# ── Streamlit page setup ──────────────────────────────
st.set_page_config(page_title="Bulldozer Price Prediction", page_icon="🚜")
st.title("🚜 Bulldozer Price Prediction")
st.markdown("Upload a CSV with bulldozer sales data; the model will predict SalePrice for you.")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

if uploaded_file:
    try:
        # 1️⃣ Read uploaded file
        df_display = pd.read_csv(uploaded_file, low_memory=False, parse_dates=['saledate'])
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df_display.head(5))

        # 2️⃣ Add date features using ingestion logic
        df_processed = data_ingestion.add_date_features(df_display.copy())

        # 3️⃣ Drop SalePrice if exists
        if "SalePrice" in df_processed.columns:
            X_processed = df_processed.drop(columns=["SalePrice"])
        else:
            X_processed = df_processed

        # 4️⃣ Load preprocessor
        preprocessor_path = trans_cfg.preprocessor
        if not os.path.exists(preprocessor_path):
            st.error(f"❌ Preprocessor file not found at: {preprocessor_path}")
            st.stop()
        preprocessor = joblib.load(preprocessor_path)

        # 5️⃣ Transform features
        X_final = preprocessor.transform(X_processed)

        # 6️⃣ Load trained model
        model_path = config['model_tuner']['tuner_save_path']
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.stop()
        model = joblib.load(model_path)

        # 7️⃣ Predict sale prices
        preds = model.predict(X_final)

        # 8️⃣ Attach predictions to original DF
        df_display['Predicted SalePrice'] = preds

        # 9️⃣ Show predictions
        st.subheader("🔍 Prediction Results")
        st.dataframe(df_display[['saledate', 'Predicted SalePrice']].head(10))

        # 🔟 Offer download button
        csv_out = df_display.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV with Predictions",
                           data=csv_out,
                           file_name="bulldozer_predictions.csv",
                           mime="text/csv")

        # ✅ Debug paths if needed
        st.write(f"ℹ️ Using preprocessor: `{preprocessor_path}`")
        st.write(f"ℹ️ Using model: `{model_path}`")

    except Exception as e:
        st.error(f"❌ Processing error: {e}")
