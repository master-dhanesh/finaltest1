from fastapi import FastAPI, UploadFile, File, Response, Request
from PIL import Image
import io

from src.inference import predict

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import time

from src.metrics import REQUEST_COUNT, REQUEST_LATENCY, PREDICTION_COUNT
from src.prediction_log import log_prediction


app = FastAPI(title="Fruit Freshness API", version="1.0.0")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code,
    ).inc()

    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)

    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    result = predict(image)
    PREDICTION_COUNT.labels(label=result["label"]).inc()
    log_prediction(file.filename, result["label"], result["confidence"])

    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
