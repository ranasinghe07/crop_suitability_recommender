import joblib
import pandas as pd

# Load saved files
model = joblib.load("model/crop_recommender_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

def predict_crop(input_dict):
    """
    input_dict: dictionary with keys like 
        {"N": 90, "P": 40, "K": 40, "temperature": 25, "humidity": 60, "ph": 6.5, "rainfall": 100}
    """
    df = pd.DataFrame([input_dict])
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)
    crop = label_encoder.inverse_transform(prediction)[0]
    return crop

if __name__ == "__main__":
    sample = {"N": 90, "P": 40, "K": 40, "temperature": 25, "humidity": 60, "ph": 6.5, "rainfall": 100}
    print("Predicted Crop:", predict_crop(sample))
