FROM python:3.13-slim

# OSMesa gives MuJoCo an offscreen OpenGL context without a GPU or a display.
# g++ builds the C++ flight core.
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglib2.0-0 g++ \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1

WORKDIR /work
COPY requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

CMD ["python", "-m", "pytest", "-q"]
