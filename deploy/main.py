from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from deploy.inference import predict

app = FastAPI(title="DR Detection API")

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
async def predict_dr(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return predict(image)
