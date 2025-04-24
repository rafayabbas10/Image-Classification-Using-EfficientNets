from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI(title="Image Classification API using EfficientNetB0")

# Load the model
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = tf.keras.models.load_model('best_model.keras')

def preprocess_image(image_bytes):
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(image_bytes))
    
    # Convert PIL Image to numpy array
    image = np.array(image)
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image = cv2.resize(image, (224, 224))
    
    # Normalize pixel values
    image = image.astype('float32') / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        
        # Preprocess the image
        image = preprocess_image(contents)
        
        # Make prediction
        predictions = model.predict(image)
        
        # Get the predicted class
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # Map class index to label (modify these labels according to your model)
        class_labels = {0: "ai", 1: "real"}
        predicted_label = class_labels.get(predicted_class, "unknown")
        
        return JSONResponse({
            "success": True,
            "prediction": {
                "class": predicted_label,
                "confidence": confidence
            }
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500) 