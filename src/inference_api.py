"""
API FastAPI que sirve el modelo STS Cross-encoder + LoRA.

Sigue el patrón visto en clase (slides 36 y 62 del tema 2):
  - FastAPI con lifespan para cargar el modelo en arranque.
  - Validación de inputs con Pydantic.
  - Predicción y retorno JSON.
  - Manejo adecuado de errores.
  - Logging de inputs problemáticos.

Endpoints:
  GET  /          → Mensaje de bienvenida.
  GET  /health    → Healthcheck (usado por Docker/cloud).
  POST /predict   → Recibe {sentence1, sentence2}, devuelve {score, ...}.

Variables de entorno:
  MODEL_CHECKPOINT_PATH: ruta al .ckpt a cargar (default: models/best_model.ckpt).
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import STSPredictor

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuración: ruta del checkpoint vía env var (clave para Docker)
# -----------------------------------------------------------------------------
CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", "models/best_model.ckpt")


# -----------------------------------------------------------------------------
# Schemas Pydantic (input/output validados)
# -----------------------------------------------------------------------------
class STSInput(BaseModel):
    """Input del endpoint /predict: par de frases a comparar."""
    sentence1: str = Field(
        ..., min_length=1, description="Primera frase",
        example="A man is playing a guitar.",
    )
    sentence2: str = Field(
        ..., min_length=1, description="Segunda frase",
        example="A guy is playing the guitar.",
    )


class STSOutput(BaseModel):
    """Output del endpoint /predict: score de similitud."""
    score: float = Field(..., description="Score de similitud en [0, 5]")
    model: str = Field(..., description="Modelo utilizado")
    sentence1: str
    sentence2: str


class HealthOutput(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool


# -----------------------------------------------------------------------------
# Lifespan: carga el modelo al arrancar la API (patrón visto en clase)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("==================== ARRANCANDO ====================")
    logger.info(f"Cargando checkpoint desde: {CHECKPOINT_PATH}")

    # Cargamos el predictor (modelo + tokenizer) UNA sola vez
    predictor = STSPredictor(checkpoint_path=CHECKPOINT_PATH)
    app.state.predictor = predictor

    logger.info("Modelo cargado. API lista para recibir peticiones.")
    yield  # <- aquí la API queda disponible

    # Teardown
    logger.info("==================== ADIOS ====================")
    app.state.predictor = None


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="STS - Similitud Semántica",
    description=(
        "API que estima la similitud semántica entre pares de frases. "
        "Modelo: Cross-encoder RoBERTa + LoRA entrenado sobre STS Benchmark."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    """Mensaje de bienvenida + enlaces útiles."""
    return {
        "message": "Hola, esta es la aplicación de MLOps - STS Similarity",
        "model": "Cross-encoder RoBERTa + LoRA",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthOutput)
def health():
    """Healthcheck: la API vive y el modelo está cargado."""
    return HealthOutput(
        status="ok",
        model_loaded=app.state.predictor is not None,
    )


@app.post("/predict", response_model=STSOutput)
def predict(data: STSInput):
    """
    Predice el score de similitud entre dos frases (rango 0-5).
    """
    if app.state.predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        score = app.state.predictor.predict(data.sentence1, data.sentence2)
    except (TypeError, ValueError) as e:
        # Input mal formado → 400 + log de input problemático
        logger.warning(
            f"Input problemático: s1={data.sentence1!r}, s2={data.sentence2!r}, error={e}"
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Cualquier otro error → 500 (no filtramos internals al usuario)
        logger.exception("Error inesperado en /predict")
        raise HTTPException(status_code=500, detail="Error interno al inferir")

    return STSOutput(
        score=round(score, 4),
        model="exp4-crossencoder-lora",
        sentence1=data.sentence1,
        sentence2=data.sentence2,
    )
