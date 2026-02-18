# =========================
# Multi-stage Dockerfile for RAG Pipeline
# =========================

# -------------------------
# Stage 1: Build Python dependencies
# ---------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies needed to build Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages into /install 
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# -------------------------
# Stage 2: Final runtime image
# -------------------------
FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/bin/ffmpeg:${PATH}"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    ffmpeg \
    poppler-utils \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder 
COPY --from=builder /install /usr/local

# Copy the rest of your application
COPY . .

# Download NLTK data
RUN python -m nltk.downloader punkt punkt_tab


CMD ["python", "rag_pipeline.py"]