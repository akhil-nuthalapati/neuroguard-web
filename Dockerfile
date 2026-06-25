FROM python:3.11-slim

# All system deps MediaPipe + OpenCV need (force fresh layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libgomp1 \
    libgles2 \
    libegl1 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# Force software rendering so MediaPipe works without GPU/display
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=4.5
ENV GALLIUM_DRIVER=softpipe
ENV MEDIAPIPE_DISABLE_GPU=1
ENV DISPLAY=:0

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Writable dirs
RUN mkdir -p data reports

EXPOSE 8080

CMD ["python", "app.py"]
