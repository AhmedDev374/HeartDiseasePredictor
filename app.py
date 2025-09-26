import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .healthy {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Use absolute paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "saved_model.joblib"

# CORRECTED MAPPINGS - Based on your model's expected encoding
# From debug: Model expects (25, 0, 0, 110, 160, 0, 0, 180, 0, 0.0, 1, 0, 3)
CP_MAPPING = {
    "Typical Angina": 0,  # Model expects 0 for baseline
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

RESTECG_MAPPING = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

SLOPE_MAPPING = {
    "Upsloping": 1,  # Model expects 1 for baseline (NOT 0)
    "Flat": 0,  # Swapped!
    "Downsloping": 2
}

THAL_MAPPING = {
    "Normal": 3,  # Model expects 3 for baseline (NOT 1)
    "Fixed Defect": 2,  # Swapped!
    "Reversible Defect": 1  # Swapped!
}


@st.cache_resource
def load_model():
    """Load the trained model package."""
    try:
        if MODEL_PATH.exists():
            return joblib.load(MODEL_PATH)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def predict_heart_disease(model, scaler, feature_names, input_data):
    """Predict heart disease for input data."""
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    return pred, proba


def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>',
                unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading AI model..."):
        model_data = load_model()

    if model_data is None:
        st.error("""
        ❌ **Model not available!**

        Please run `main.py` first to train the model, then refresh this page.
        """)
        return

    # Extract components
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    train_accuracy = model_data.get('train_accuracy', 0)
    test_accuracy = model_data.get('test_accuracy', 0)
    heart_data = model_data['heart_data']
    baseline_tuple = model_data.get('baseline_tuple', (25, 0, 0, 110, 160, 0, 0, 180, 0, 0.0, 1, 0, 3))

    # Sidebar
    with st.sidebar:
        st.header("Model Information")
        st.metric("Training Accuracy", f"{train_accuracy * 100:.2f}%")
        st.metric("Test Accuracy", f"{test_accuracy * 100:.2f}%")
        st.info("This model predicts heart disease risk based on 13 medical parameters.")

        if st.checkbox("Show Sample Data"):
            st.dataframe(heart_data.head(10))

        debug_mode = st.checkbox("Show encoding debug", value=True)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Patient Information")
        st.subheader("Enter Medical Parameters")

        with st.form("patient_form"):
            # Demographic information - DEFAULT: Female (0), Age 25
            st.write("### Demographic Information")
            age = st.slider("Age", 20, 100, 25)  # Default: 25
            sex = st.radio("Sex", ["Female", "Male"], index=0)  # Default: Female (0)
            sex_encoded = 0 if sex == "Female" else 1

            # Medical history - DEFAULTS to match baseline tuple
            st.write("### Medical History")
            cp = st.selectbox(
                "Chest Pain Type",
                list(CP_MAPPING.keys()),
                index=0  # Default: Typical Angina (0) - MODEL EXPECTS 0!
            )
            cp_encoded = CP_MAPPING[cp]

            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 110)  # Default: 110
            chol = st.slider("Cholesterol (mg/dl)", 100, 600, 160)  # Default: 160

            # Additional parameters - DEFAULTS to match baseline tuple
            st.write("### Additional Parameters")
            fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], index=0)  # Default: No (0)
            fbs_encoded = 1 if fbs == "Yes" else 0

            restecg = st.selectbox(
                "Resting Electrocardiographic Results",
                list(RESTECG_MAPPING.keys()),
                index=0  # Default: Normal (0)
            )
            restecg_encoded = RESTECG_MAPPING[restecg]

            thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 180)  # Default: 180
            exang = st.radio("Exercise Induced Angina", ["No", "Yes"], index=0)  # Default: No (0)
            exang_encoded = 1 if exang == "Yes" else 0

            oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 0.0, 0.1)  # Default: 0.0

            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                list(SLOPE_MAPPING.keys()),
                index=0  # Default: Upsloping (1) - MODEL EXPECTS 1!
            )
            slope_encoded = SLOPE_MAPPING[slope]

            ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)  # Default: 0

            thal = st.selectbox(
                "Thalassemia",
                list(THAL_MAPPING.keys()),
                index=0  # Default: Normal (3) - MODEL EXPECTS 3!
            )
            thal_encoded = THAL_MAPPING[thal]

            submitted = st.form_submit_button("Predict Heart Disease Risk")

    with col2:
        st.header("Prediction Results")

        if submitted:
            # Build input tuple in correct feature order
            input_data = (
                age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded,
                restecg_encoded, thalach, exang_encoded, oldpeak,
                slope_encoded, ca, thal_encoded
            )

            # Debug information
            if debug_mode:
                with st.expander("🔍 Debug Information", expanded=True):
                    st.write("**Feature names:**", feature_names)
                    st.write("**Input data (encoded):**", input_data)
                    st.write("**Baseline tuple:**", baseline_tuple)
                    st.write("**Match:**", input_data == baseline_tuple)

                    if input_data != baseline_tuple:
                        st.write("**Mismatches:**")
                        for i, (a, b) in enumerate(zip(input_data, baseline_tuple)):
                            if a != b:
                                st.write(f"  Position {i}: {a} vs {b} (DIFFERENT!)")

                    st.write("**Mappings used:**")
                    st.write("Chest Pain:", CP_MAPPING)
                    st.write("Resting ECG:", RESTECG_MAPPING)
                    st.write("Slope:", SLOPE_MAPPING)
                    st.write("Thalassemia:", THAL_MAPPING)

            # Make prediction
            try:
                prediction, probability = predict_heart_disease(
                    model, scaler, feature_names, input_data
                )

                # Display results
                risk_percentage = probability[1] * 100
                st.subheader("Prediction Result")

                if risk_percentage >= 50:
                    st.markdown(
                        f'<div class="prediction-box risk">'
                        f'<h2>⚠️ HIGH RISK</h2>'
                        f'<p>The person may have heart disease</p>'
                        f'<p>Risk Probability: {risk_percentage:.1f}%</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box healthy">'
                        f'<h2>✅ LOW RISK</h2>'
                        f'<p>The person is unlikely to have heart disease</p>'
                        f'<p>Risk Probability: {risk_percentage:.1f}%</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # Probability chart
                st.subheader("Risk Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Condition': ['No Heart Disease', 'Heart Disease'],
                    'Probability': [probability[0] * 100, probability[1] * 100]
                })
                st.bar_chart(prob_df.set_index('Condition'))

                # Input summary
                with st.expander("View Input Summary"):
                    input_summary = pd.DataFrame({
                        'Parameter': [
                            'Age', 'Sex', 'Chest Pain', 'Blood Pressure', 'Cholesterol',
                            'Blood Sugar', 'Resting ECG', 'Max Heart Rate', 'Exercise Angina',
                            'ST Depression', 'Slope', 'Major Vessels', 'Thalassemia'
                        ],
                        'Value': [
                            f"{age} years", sex, cp, f"{trestbps} mm Hg", f"{chol} mg/dl",
                            fbs, restecg, f"{thalach} bpm", exang, f"{oldpeak:.1f}",
                            slope, f"{ca} vessels", thal
                        ]
                    })
                    st.table(input_summary)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                if debug_mode:
                    st.exception(e)

        else:
            # Default message when form not submitted
            st.info("Please fill out the form on the left and click 'Predict Heart Disease Risk' to see results.")
            st.image("https://images.unsplash.com/photo-1582719471384-894fbb16e074?w=400",
                     caption="Heart Health Assessment")

    # Footer
    st.markdown("---")
    st.caption(
        "⚠️ This tool is for educational purposes only. Please consult healthcare professionals for medical advice.")


if __name__ == "__main__":
    main()