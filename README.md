# Image Classification API using EfficientNetB0

This FastAPI application serves an EfficientNetB0 model trained to classify images as either AI-generated or real.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the model file `best_model.keras` in the root directory.

3. Run the API:
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /predict

Upload an image to get the classification prediction.

**Request:**
- Method: POST
- Endpoint: `/predict`
- Content-Type: multipart/form-data
- Body: file (image file)

**Response:**
```json
{
    "success": true,
    "prediction": {
        "class": "ai",
        "confidence": 0.95
    }
}
```

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI). 