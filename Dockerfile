# 1. Start with a solid Python base image. 
# (Since your requirements list torch==2.10.0, pip will automatically pull the 
# correct PyTorch version with bundled CUDA libraries, so a standard Python image is great).
FROM python:3.11-slim

# 2. Set environment variables to keep Python behavior predictable 
# and tell MuJoCo to use EGL for headless hardware acceleration (or OSMesa for CPU)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=egl

# 3. Set the working directory
WORKDIR /workspace

# 4. Install system-level dependencies required by MuJoCo
# Even headless, MuJoCo needs these to load the physics environment without crashing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-dev \
    libgl1 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy your requirements file into the container
COPY requirements.txt .

# 6. Install your Python dependencies
# We upgrade pip first to ensure smooth installation of modern packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
