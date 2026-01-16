from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io

from deploy.inference import predict

app = FastAPI(title="DR Detection API")

templates = Jinja2Templates(directory="templates")

# -------------------------
# UI
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# -------------------------
# UI Prediction
# -------------------------
@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request, file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    result = predict(image)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": f"{result['prediction']} (DR prob: {result['dr_probability']})"
        }
    )

# -------------------------
# API Prediction (JSON)
# -------------------------
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return predict(image)

# -------------------------
# Health Check
# -------------------------
@app.get("/health")
def health():
    return {"status": "running"}
