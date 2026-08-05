# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for compiling if any (slim doesn't have build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Create cache directory inside container
RUN mkdir -p data/cache

# Pre-train the model during build so startup is instant in production
RUN python -c "from app import get_or_train_model; get_or_train_model()"

# Expose port (Cloud Run will inject PORT environment variable, app.py is set to bind to it)
EXPOSE 8080

# Command to run Flask app
CMD ["python", "app.py"]
