import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_datasets():
    crop_df = pd.read_csv("data/Crop_recommendation.csv")
    location_df = pd.read_csv("data/locationData.csv")
    weather_df = pd.read_csv("data/weatherData.csv")
    climate_df = pd.read_csv("data/Sri_Lanka_Climate_Data.csv")
    return crop_df, location_df, weather_df, climate_df

def preprocess_data(crop_df, location_df, weather_df, climate_df):
    # For simplicity, merge only on location_id if available
    if "location_id" in crop_df.columns and "location_id" in location_df.columns:
        merged = crop_df.merge(location_df, on="location_id", how="left")
    else:
        merged = crop_df.copy()

    # Fill missing values
    for col in merged.columns:
        if merged[col].dtype in ["float64", "int64"]:
            merged[col].fillna(merged[col].mean(), inplace=True)
        else:
            merged[col].fillna(merged[col].mode()[0], inplace=True)

    # Encode labels
    label_encoder = LabelEncoder()
    merged["label"] = label_encoder.fit_transform(merged["label"])

    # Features and target
    X = merged.drop(columns=["label"])
    y = merged["label"]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoder
