# ─────────────────────────────────────────────
#  STAGE 1: builder — instala dependencias pesadas
# ─────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /install

# Dependencias del sistema necesarias para OpenCV, pydicom, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (cache de Docker)
COPY requirements.txt .

# Instalar todo en /install/packages para copiar en la imagen final
RUN pip install --upgrade pip && \
    pip install --prefix=/install/packages --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
#  STAGE 2: runtime — imagen final liviana
# ─────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Solo librerías de sistema en runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar paquetes Python desde builder
COPY --from=builder /install/packages /usr/local

# Copiar código fuente del backend
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/uploads /app/models /app/data

# Variables de entorno por defecto (sobreescribibles en docker-compose)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATABASE_URL=sqlite:///./data/lung_detection.db \
    SECRET_KEY=change-me-in-production \
    ACCESS_TOKEN_EXPIRE_MINUTES=60 \
    UPLOAD_DIR=/app/uploads \
    MODELS_DIR=/app/models

EXPOSE 8000

# Healthcheck — FastAPI responde en /docs o en /
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Arrancar con uvicorn en modo producción (sin --reload)
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]