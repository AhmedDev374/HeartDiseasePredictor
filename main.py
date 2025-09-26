import sys
import subprocess
from pathlib import Path
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

# Use absolute paths to avoid confusion
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "saved_model.joblib"
APP_PATH = BASE_DIR / "app.py"
DATA_PATH = BASE_DIR / "heart_disease_data.csv"

# Baseline tuple for prediction - CORRECT ENCODING
BASELINE_TUPLE = (25, 0, 0, 110, 160, 0, 0, 180, 0, 0.0, 1, 0, 3)


def analyze_features(heart_data):
    """Analyze feature distributions and correlations"""
    print("\n🔍 Feature Analysis:")
    print("Target distribution:")
    print(heart_data['target'].value_counts())

    print("\nHigh-risk cases analysis:")
    high_risk = heart_data[heart_data['target'] == 1]
    low_risk = heart_data[heart_data['target'] == 0]

    print(f"High-risk cases: {len(high_risk)}")
    print(f"Low-risk cases: {len(low_risk)}")

    # Analyze blood pressure in high-risk vs low-risk
    print("\nBlood Pressure analysis:")
    print(f"High-risk avg BP: {high_risk['trestbps'].mean():.1f}")
    print(f"Low-risk avg BP: {low_risk['trestbps'].mean():.1f}")
    print(f"High-risk max BP: {high_risk['trestbps'].max()}")
    print(f"Low-risk max BP: {low_risk['trestbps'].max()}")


def train_heart_disease_model(save_path: Path = None):
    """Train heart disease model with better parameters."""
    if save_path is None:
        save_path = MODEL_PATH

    print("Loading training data...")
    heart_data = pd.read_csv(DATA_PATH)

    # Analyze features
    analyze_features(heart_data)

    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    feature_names = X.columns.tolist()

    print(f"Feature names: {feature_names}")

    print("Splitting data...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42
    )

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training XGBoost model with better parameters...")
    num_pos = int(sum(Y_train))
    num_neg = len(Y_train) - num_pos
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    # Use more conservative parameters to prevent overfitting
    model = XGBClassifier(
        n_estimators=100,  # Reduced to prevent overfitting
        max_depth=4,  # Slightly deeper for better feature interactions
        learning_rate=0.05,  # Slower learning
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,  # Stronger regularization
        reg_lambda=2,  # Stronger regularization
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train_scaled, Y_train)

    # Feature importance
    print("\n📊 Feature Importance:")
    importance_scores = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    print(feature_importance)

    print("Calibrating model...")
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)  # More folds
    calibrated_model.fit(X_train_scaled, Y_train)

    # Predictions
    train_pred = calibrated_model.predict(X_train_scaled)
    test_pred = calibrated_model.predict(X_test_scaled)

    # Create comprehensive results package
    results = {
        'model': calibrated_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'train_accuracy': accuracy_score(Y_train, train_pred),
        'test_accuracy': accuracy_score(Y_test, test_pred),
        'heart_data': heart_data,
        'class_report': classification_report(Y_test, test_pred),
        'confusion_matrix': confusion_matrix(Y_test, test_pred),
        'baseline_tuple': BASELINE_TUPLE,
        'feature_importance': feature_importance
    }

    print(f"\n📈 Model Performance:")
    print(f"Training Accuracy: {results['train_accuracy'] * 100:.2f}%")
    print(f"Test Accuracy: {results['test_accuracy'] * 100:.2f}%")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")

    print(f"Saving model to {save_path}...")
    dump(results, save_path)
    print("✅ Model trained and saved successfully!")

    return results


def predict_heart_disease(model, scaler, feature_names, input_data):
    """Predict heart disease for input data."""
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    return pred, proba


def test_extreme_cases(model, scaler, feature_names):
    """Test if model responds properly to extreme values"""
    print("\n🧪 Testing Extreme Cases:")

    # Test very high blood pressure
    extreme_bp = list(BASELINE_TUPLE)
    extreme_bp[3] = 200  # Change blood pressure to 200
    extreme_bp = tuple(extreme_bp)

    pred, proba = predict_heart_disease(model, scaler, feature_names, extreme_bp)
    print(f"BP=200: Prediction={pred}, Risk={proba[1] * 100:.1f}%")

    # Test very high cholesterol
    extreme_chol = list(BASELINE_TUPLE)
    extreme_chol[4] = 400  # Change cholesterol to 400
    extreme_chol = tuple(extreme_chol)

    pred, proba = predict_heart_disease(model, scaler, feature_names, extreme_chol)
    print(f"Chol=400: Prediction={pred}, Risk={proba[1] * 100:.1f}%")


if __name__ == "__main__":
    print("=== Heart Disease Prediction System ===")

    # 🚨 ALWAYS TRAIN A NEW MODEL - DON'T USE EXISTING ONE
    print("🚨 Training new model (existing model will be overwritten)...")

    # Remove existing model if it exists
    if MODEL_PATH.exists():
        print("🗑️ Removing existing model...")
        MODEL_PATH.unlink()

    # Train new model
    results = train_heart_disease_model()

    model = results['model']
    scaler = results['scaler']
    feature_names = results['feature_names']

    # 2️⃣ Predict for baseline tuple
    print("\n🧪 Testing baseline prediction...")
    pred, proba = predict_heart_disease(model, scaler, feature_names, BASELINE_TUPLE)
    print("Baseline tuple:", BASELINE_TUPLE)
    print("Predicted class:", "LOW RISK (0)" if pred == 0 else "HIGH RISK (1)")
    print(f"Probability → No Disease: {proba[0] * 100:.2f}% | Disease: {proba[1] * 100:.2f}%")

    # 3️⃣ Test extreme cases
    test_extreme_cases(model, scaler, feature_names)

    # 4️⃣ Launch Streamlit app
    print(f"\n🚀 Launching Streamlit app from {APP_PATH}...")
    if APP_PATH.exists():
        try:
            cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n👋 App closed by user")
        except Exception as e:
            print(f"❌ Error launching app: {e}")
    else:
        print(f"❌ Cannot find app.py at {APP_PATH}")