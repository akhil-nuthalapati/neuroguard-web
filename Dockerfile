FROM python:3.11-slim

# Only what OpenCV headless needs (no GL/GPU at all)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data reports

EXPOSE 8080

CMD ["python", "app.py"]
