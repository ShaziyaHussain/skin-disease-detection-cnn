# Skin Disease Detection using CNN (Deployed on Google Cloud Run)

This project is an AI-powered skin disease classification system built using Convolutional Neural Networks (CNN) and deployed as a scalable API on Google Cloud Run.

## Features
- Detects 6 types of skin conditions: Bullous, Eczema, Melanoma, Nail Fungus, Normal Skin, Vascular Tumors
- TensorFlow-based CNN model trained in Google Colab
- Flask API deployed on Cloud Run
- Docker containerization
- Simple REST API for predictions

## Project Structure
```
skin-disease-api/
│── app.py
│── requirements.txt
│── Dockerfile
│── best_model.h5
│── README.md
```

## API Endpoints
### Health Check
GET /

### Predict Disease
POST /predict with `file` as uploaded image.

## Deploy on Cloud Run
```
gcloud run deploy skin-disease-api --source .
```
