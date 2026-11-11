# ===== 1. Base image =====
# A minimal Linux image with Python 3.11 already installed.
FROM python:3.11-slim


# ===== 2. Working directory =====
# Created automatically if it does not exist.
WORKDIR /app


# ===== 3. Install dependencies =====
# Copy only the requirements file first.
# Docker caches this layer as long as requirements.txt does not change.
# This avoids reinstalling dependencies every time the application changes.

COPY requirements.txt .
RUN pip install -r requirements.txt


# ===== 4. Copy application code and model =====
# These files change more frequently than requirements.txt.

COPY app.py .
COPY model/ model/


# ===== 5. Set the default model location =====
# The API will load the model from the local model/ folder.
ENV MODEL_URI=model


# ===== 6. Expose the API port =====
# Documentation only; does not publish the port automatically.
EXPOSE 8000


# ===== 7. Start the FastAPI application =====
# --host 0.0.0.0 allows connections from outside the container.

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]