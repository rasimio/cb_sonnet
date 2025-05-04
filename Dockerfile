# Use Python 3.11.9 as the base image
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install poetry
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir wheel setuptools

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories
RUN mkdir -p data logs saved_models backtest_results

# Expose port for the API
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]