import os
import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
from google.cloud import storage

app = Flask(__name__)

# -----------------------------
# Download model from Cloud Storage
# -----------------------------
def download_model_from_gcs():
    bucket_name = os.environ.get("BUCKET_NAME")
    model_blob_name = os.environ.get("MODEL_FILE")

    if not bucket_name or not model_blob_name:
        raise Exception("Environment variables BUCKET_NAME and MODEL_FILE are required")

    local_model_path = "/tmp/best_model.h5"   # Cloud Run allows writing to /tmp only

    # Download from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_blob_name)
    blob.download_to_filename(local_model_path)

    print(f"Model downloaded from gs://{bucket_name}/{model_blob_name} → {local_model_path}")
    return local_model_path


print("Downloading model from Google Cloud Storage...")
MODEL_PATH = download_model_from_gcs()
print("Loading TensorFlow model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


# Your class labels
classes = [
    'Bullous',
    'Eczema',
    'Melanoma',
    'Nail Fungus',
    'Normal skin',
    'Vascular Tumors'
]


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["file"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)

        pred = model.predict(arr)
        predicted_class = classes[int(np.argmax(pred))]
        confidence = float(np.max(pred))

        return jsonify({
            "class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Health Check Route
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Skin Disease Detection API is running ✔", 200


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
