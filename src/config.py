"""
Configuración centralizada del proyecto STS-MLOps.

Todos los hiperparámetros están aquí para que sean fácilmente parametrizables
y trazables desde W&B / CLI / Docker. Esta es la única fuente de verdad.
"""
from peft import LoraConfig

# -----------------------------------------------------------------------------
# Modelo y datos
# -----------------------------------------------------------------------------
MODEL_NAME = "FacebookAI/roberta-base"
DATASET_NAME = "mteb/stsbenchmark-sts"

# -----------------------------------------------------------------------------
# Hiperparámetros del Experimento 4 (Cross-encoder + LoRA) - el ganador
# -----------------------------------------------------------------------------
MAX_LENGTH = 128
BATCH_SIZE = 32
POOLING = "mean"
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 3

OPTIMIZER_PARAMS = {
    "lr": 2e-5,
    "weight_decay": 0.01,
}

# Configuración LoRA (función para evitar problemas de import al serializar)
def get_lora_config() -> LoraConfig:
    """Devuelve la configuración LoRA del Exp 4."""
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )

# -----------------------------------------------------------------------------
# Normalización de scores (el dataset usa [0, 5], el modelo trabaja en [0, 1])
# -----------------------------------------------------------------------------
SCORE_MIN = 0.0
SCORE_MAX = 5.0

# -----------------------------------------------------------------------------
# Reproducibilidad
# -----------------------------------------------------------------------------
SEED = 42

# -----------------------------------------------------------------------------
# Rutas
# -----------------------------------------------------------------------------
MODELS_DIR = "models"
LOGS_DIR = "logs"

# -----------------------------------------------------------------------------
# W&B
# -----------------------------------------------------------------------------
WANDB_PROJECT = "sts-mlops"
WANDB_ENTITY = "anacantalapiedraarellano-lab"  # Se rellena con el username de W&B (o se deja None)
