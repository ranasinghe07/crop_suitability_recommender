import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_datasets():
    """Load all datasets"""
    crop_df = pd.read_csv("data/Crop_recommendation.csv")
    location_df = pd.read_csv("data/locationData.csv")
    weather_df = pd.read_csv("data/weatherData.csv")
    climate_df = pd.read_csv("data/Sri_Lanka_Climate_Data.csv")
    return crop_df, location_df, weather_df, climate_df


def preprocess_data(crop_df, location_df, weather_df, climate_df):
    """Merge datasets + preprocess features/labels"""

    # For simplicity, keep only crop dataset features
    # (you can extend merging logic if district/location matches later)
    df = crop_df.copy()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode target labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    # Feature-target split
    X = df.drop(columns=["label"])
    y = df["label"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoder


def train():
    """Train models and save the best one"""

    # --- Load + preprocess ---
    crop_df, location_df, weather_df, climate_df = load_datasets()
    X, y, scaler, label_encoder = preprocess_data(crop_df, location_df, weather_df, climate_df)

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # --- KNN ---
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)

    # --- Select Best Model ---
    if acc_rf >= acc_knn:
        best_model = rf
        print("✅ Selected Model: Random Forest")
        print("Accuracy:", acc_rf)
        print(classification_report(y_test, y_pred_rf))
    else:
        best_model = knn
        print("✅ Selected Model: KNN")
        print("Accuracy:", acc_knn)
        print(classification_report(y_test, y_pred_knn))

    # --- Save all artifacts together ---
    joblib.dump(best_model, "model/crop_recommender_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(label_encoder, "model/label_encoder.pkl")

    print("🎉 Training complete. Model + Scaler + Encoder saved in /model/")


if __name__ == "__main__":
    train()
