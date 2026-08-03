from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import shutil
import os
from features import extract_features
from model import VocalShieldModel

# 1. FastAPI App Initialization
app = FastAPI(title="VocalShield Deepfake Detection API")

# 2. CORS Middleware (Browser & Swagger UI blocking fix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load ML Model
model = VocalShieldModel()
model.eval()

# 4. Root Endpoint (Health Check)
@app.get("/")
def home():
    return {"message": "VocalShield API is Running!"}

# 5. Prediction Endpoint
@app.post("/predict")
async def predict_voice(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    
    # Save uploaded audio file temporarily
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract audio features
    features = extract_features(temp_path)
    
    # Cleanup temporary file safely
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    # Validation check for extracted features
    if features is None:
        return {"status": "error", "message": "Invalid Audio File"}
        
    # Model inference
    tensor_in = torch.tensor(features).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        output = model(tensor_in)
        probs = torch.softmax(output, dim=1)
        fake_score = float(probs[0][1] * 100)
        
    # Final Structured Response
    return {
        "status": "success",
        "fake_probability": round(fake_score, 2),
        "prediction": "Deepfake" if fake_score > 50 else "Real"
    }

