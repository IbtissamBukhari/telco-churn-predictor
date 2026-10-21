from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import BatchPrediction, BatchRequest, Customer, Prediction
from src.predictor import get_predictor

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(_app):
    get_predictor()  # load the model once at startup
    yield


app = FastAPI(
    title="Telco Churn Predictor API",
    description=(
        "Predicts the probability that a telecom customer will cancel their "
        "subscription, explains which inputs drive the score, and suggests "
        "retention actions."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return get_predictor().metrics


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return get_predictor().predict(customer.model_dump())


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(request: BatchRequest):
    predictor = get_predictor()
    return {"predictions": [predictor.predict(c.model_dump()) for c in request.customers]}
