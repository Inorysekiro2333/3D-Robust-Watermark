## Allow overriding the base image so you can point to an Alibaba Cloud mirror or other registry.
ARG BASE_IMAGE=python:3.9-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies to allow some packages to compile if needed
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure storage directories exist
RUN mkdir -p static/uploads static/results/watermarked_models static/results/attacted_models \
    && chown -R root:root /app

EXPOSE 8000

# Run with a production-ready ASGI server (uvicorn). Adjust --workers to suit your CPU.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
