
import os
import json
import joblib
import pandas as pd
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ============================================
# CONFIGURATION
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_basic_pipeline.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "house_price_basic_pipeline_metadata.json")


def patch_simple_imputer_fill_dtype(obj):
    """
    Compatibility patch for old saved sklearn SimpleImputer objects.

    Sometimes a model saved with one sklearn version may fail
    in another sklearn version because SimpleImputer expects
    an internal attribute called _fill_dtype.
    """
    
    if isinstance(obj, SimpleImputer):
        if not hasattr(obj, "_fill_dtype"):
            statistics = getattr(obj, "statistics_", None)
            
            if statistics is not None and hasattr(statistics, "dtype"):
                obj._fill_dtype = statistics.dtype
            else:
                obj._fill_dtype = object

    elif isinstance(obj, Pipeline):
        for _, step in obj.steps:
            patch_simple_imputer_fill_dtype(step)

    elif isinstance(obj, ColumnTransformer):
        for _, transformer, _ in obj.transformers_:
            if transformer not in ["drop", "passthrough"]:
                patch_simple_imputer_fill_dtype(transformer)

    return obj


# ============================================
# LOAD MODEL AND METADATA
# ============================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    model = patch_simple_imputer_fill_dtype(model)
    return model


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return metadata


model = load_model()
metadata = load_metadata()
features_used = metadata["features_used"]


# ============================================
# DASHBOARD TITLE
# ============================================

st.set_page_config(
    page_title="House Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction Dashboard")

st.write(
    """
    This dashboard uses a saved machine learning pipeline to predict house prices.
    The pipeline includes preprocessing and a trained Linear Regression model.
    """
)


# ============================================
# SIDEBAR MODEL INFORMATION
# ============================================

st.sidebar.header("Model Information")

st.sidebar.write("Model name:", metadata["model_name"])
st.sidebar.write("Target:", metadata["target"])
st.sidebar.write("Model type:", metadata["model_type"])

st.sidebar.subheader("Model Metrics")
st.sidebar.write("MAE:", metadata["metrics"]["MAE"])
st.sidebar.write("RMSE:", metadata["metrics"]["RMSE"])
st.sidebar.write("R²:", metadata["metrics"]["R2"])


# ============================================
# TABS
# ============================================

tab1, tab2 = st.tabs(["Single Prediction", "Batch CSV Prediction"])


# ============================================
# TAB 1: SINGLE HOUSE PREDICTION
# ============================================

with tab1:
    st.header("Single House Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        suburb = st.selectbox(
            "Suburb",
            ["CBD", "Northside", "Southside", "Eastside", "Westside", "Hillside"]
        )

        property_type = st.selectbox(
            "Property Type",
            ["House", "Townhouse", "Apartment"]
        )

        condition = st.selectbox(
            "Condition",
            ["Poor", "Fair", "Good", "Excellent"]
        )

        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)

    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        parking_spaces = st.number_input("Parking Spaces", min_value=0, max_value=10, value=1)
        land_size_m2 = st.number_input("Land Size (m²)", min_value=1, value=450)
        building_size_m2 = st.number_input("Building Size (m²)", min_value=1, value=180)

    with col3:
        house_age_years = st.number_input("House Age (Years)", min_value=0, max_value=150, value=12)
        distance_to_cbd_km = st.number_input("Distance to CBD (km)", min_value=0.0, value=8.5)
        nearest_school_km = st.number_input("Nearest School (km)", min_value=0.0, value=1.2)
        nearest_station_km = st.number_input("Nearest Station (km)", min_value=0.0, value=2.0)

    single_house = pd.DataFrame({
        "suburb": [suburb],
        "property_type": [property_type],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "parking_spaces": [parking_spaces],
        "land_size_m2": [land_size_m2],
        "building_size_m2": [building_size_m2],
        "house_age_years": [house_age_years],
        "distance_to_cbd_km": [distance_to_cbd_km],
        "nearest_school_km": [nearest_school_km],
        "nearest_station_km": [nearest_station_km],
        "condition": [condition]
    })

    st.subheader("Input Preview")
    st.dataframe(single_house)

    if st.button("Predict House Price"):
        prediction = model.predict(single_house[features_used])
        predicted_price = round(prediction[0], 0)

        st.success(f"Predicted House Price: {predicted_price:,.0f}")


# ============================================
# TAB 2: BATCH CSV PREDICTION
# ============================================

with tab2:
    st.header("Batch CSV Prediction")

    st.write(
        """
        Upload a CSV file containing new house data.
        The file must contain the same feature columns used during training.
        """
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(batch_data.head())

        missing_columns = []

        for col in features_used:
            if col not in batch_data.columns:
                missing_columns.append(col)

        if len(missing_columns) > 0:
            st.error(f"The uploaded file is missing required columns: {missing_columns}")
        else:
            batch_input = batch_data[features_used]
            batch_predictions = model.predict(batch_input)

            result = batch_data.copy()
            result["predicted_price"] = batch_predictions.round(0).astype(int)

            st.subheader("Prediction Result")
            st.dataframe(result)

            csv_output = result.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Result as CSV",
                data=csv_output,
                file_name="house_price_prediction_result.csv",
                mime="text/csv"
            )
