FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY frontend ./frontend
COPY models/metrics.json ./models/metrics.json

# Download the dataset and train the model inside the image, so the model is
# always built with the same scikit-learn version that serves it.
RUN python -m src.train && rm -rf data/raw/*.csv

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Render and most hosts pass the port in $PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
