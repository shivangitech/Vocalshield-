from fastapi import FastAPI, UploadFile, File
import torch
import shutil
import os
from features import extract_features
from model import VocalShieldModel

app = FastAPI(title="VocalShield Deepfake Detection API")
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
        "result": "AI DEEPFAKE" if fake_score > 50 else "GENUINE HUMAN",
        "deepfake_probability": f"{round(fake_score, 2)}%",
        "risk_level": "HIGH" if fake_score > 75 else ("MEDIUM" if fake_score > 45 else "LOW")
    }
