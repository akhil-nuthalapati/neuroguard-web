FROM python:3.11-slim

# System deps for OpenCV headless + MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create writable directories
RUN mkdir -p data reports

# Expose port
EXPOSE 8080

# Run with gunicorn + eventlet for WebSocket support
CMD ["python", "app.py"]
