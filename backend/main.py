from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from src.inference import predict

app = FastAPI(title="Fruit Freshness API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    result = predict(image)
    return result
