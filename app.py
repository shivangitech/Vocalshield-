from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # <-- Naya add hua hai
import torch
import shutil
import os
from features import extract_features
from model import VocalShieldModel

app = FastAPI(title="VocalShield Deepfake Detection API")

# <-- Naya CORS policy add hua hai (Browser block ko bypass karne ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = VocalShieldModel()
model.eval()

@app.get("/")
def home():
    return {"message": "VocalShield API is Running!"}

@app.post("/predict")
async def predict_voice(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    features = extract_features(temp_path)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    if features is None:
        return {"error": "Invalid Audio File"}
        
    tensor_in = torch.tensor(features).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        output = model(tensor_in)
        probs = torch.softmax(output, dim=1)
        fake_score = float(probs[0][1] * 100)
        
    return {
        "status": "success",
        "fake_probability": round(fake_score, 2),
        "prediction": "Deepfake" if fake_score > 50 else "Real"
    }

