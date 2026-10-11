FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY results/fusion.json ./results/fusion.json
COPY backend ./backend
RUN mkdir -p /app/backend/models && chmod 777 /app/backend/models
COPY frontend/dist ./frontend/dist

WORKDIR /app/backend

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
