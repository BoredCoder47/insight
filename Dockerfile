FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV MEDIAPIPE_DISABLE_GPU=1
ENV EGL_PLATFORM=surfaceless
ENV LIBGL_ALWAYS_SOFTWARE=1

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
