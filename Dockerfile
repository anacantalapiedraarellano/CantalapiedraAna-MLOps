# =============================================================================
# Dockerfile para la API de STS Similarity (Cross-encoder + LoRA)
# Sigue las buenas prácticas vistas en clase (slide 23 del tema 4):
#   1. Imagen base
#   2. Dependencias del sistema
#   3. requirements.txt
#   4. pip install
#   5. Copiar código
#   6. Definir CMD
# =============================================================================

# 1. Imagen base: Python 3.10 slim (~ misma que clase, ligera)
FROM python:3.10-slim

# Variables de entorno: Python sin .pyc, output sin buffer (mejor en logs Docker)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 2. Dependencias del sistema: build-essential lo necesitan algunos paquetes
#    como tokenizers/sentencepiece para compilar wheels nativos.
#    Limpieza de cache de apt al final para reducir tamaño de la imagen.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3 y 4. Primero requirements (capa cacheable: si no cambian, no se reinstala todo)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar SÓLO lo necesario para la API (slide 24):
#    - src/ : código de producción (API + modelo + utilidades)
#    - models/: pesos entrenados que la API cargará en arranque
#    NO copiamos: data/, notebooks/, tests/, logs/, wandb/, venv/.
#    Esto se controla con .dockerignore.
COPY src/ ./src/
COPY models/ ./models/

# Variable de entorno con la ruta del checkpoint (inference_api.py la lee).
# Si quisiéramos cambiar el modelo, basta con cambiar esta variable al ejecutar.
ENV MODEL_CHECKPOINT_PATH=/app/models/best_model.ckpt

# Puerto que expone la API
EXPOSE 8000

# 6. Comando de arranque: levanta uvicorn escuchando en todas las interfaces.
#    Sin --reload (eso es de desarrollo).
CMD ["uvicorn", "src.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
