FROM python:3.11-slim


WORKDIR /app
ENV PYTHONPATH=/app/backend

# System deps for pillow/tf image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend
COPY models /app/models

ENV MODEL_PATH=/app/models/latest_model.keras
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
