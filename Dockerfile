FROM python:3.11-slim

# System deps for OpenCV headless + MediaPipe (needs EGL/GLES even headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgles2 \
    libegl1 \
    libglvnd0 \
    libglx0 \
    mesa-utils \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create writable directories
RUN mkdir -p data reports

# Force software rendering — MediaPipe needs this in headless Docker
ENV DISPLAY=:99
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MEDIAPIPE_DISABLE_GPU=1

# Expose port
EXPOSE 8080

CMD ["python", "app.py"]
