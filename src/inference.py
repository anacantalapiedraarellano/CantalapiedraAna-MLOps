"""
Lógica de inferencia para el modelo STS Cross-encoder + LoRA.

Separamos la inferencia en su propio módulo (en vez de mezclarla con la API)
para que sea fácilmente testeable y reutilizable sin necesidad de levantar
un servidor HTTP.

La clase STSPredictor implementa el patrón "carga el modelo una vez, predice
muchas". Esto es crítico para el rendimiento de la API: si cargáramos el
modelo en cada petición, cada predicción tardaría >5 segundos.
"""
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from src.config import (
    MODEL_NAME,
    MAX_LENGTH,
    POOLING,
    OPTIMIZER_PARAMS,
    get_lora_config,
)
from src.data import denormalize_score
from src.model import STSRegressorLoRA

logger = logging.getLogger(__name__)


class STSPredictor:
    """
    Predictor de similitud semántica.

    Carga el modelo y el tokenizer una sola vez al instanciarse, y expone
    el método `predict(s1, s2)` para hacer inferencias rápidas.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = MODEL_NAME,
        max_length: int = MAX_LENGTH,
        device: Optional[str] = None,
    ):
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

        # Detectar device automáticamente si no se especifica
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info(f"Cargando modelo desde {checkpoint_path} en {self.device}")
        self.model = STSRegressorLoRA.load_from_checkpoint(
            checkpoint_path,
            model_name=model_name,
            lora_config=get_lora_config(),
            optimizer_params=OPTIMIZER_PARAMS,
            pooling=POOLING,
            map_location=self.device,
        )
        self.model.eval()
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        logger.info("Modelo y tokenizer listos para inferencia")

    @torch.no_grad()
    def predict(self, sentence1: str, sentence2: str) -> float:
        """
        Predice el score de similitud entre dos frases.

        Args:
            sentence1: primera frase (string).
            sentence2: segunda frase (string).

        Returns:
            Score continuo en [0, 5]. 0 = sin relación; 5 = idénticas.
        """
        if not isinstance(sentence1, str) or not isinstance(sentence2, str):
            raise TypeError("Ambas frases deben ser strings")
        if not sentence1.strip() or not sentence2.strip():
            raise ValueError("Las frases no pueden estar vacías")

        # Tokenizamos el par concatenado (input al cross-encoder)
        encoded = self.tokenizer(
            [(sentence1, sentence2)],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        pred_norm = self.model(
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded.get("token_type_ids"),
        )  # tensor (1,) en [0, 1]

        return float(denormalize_score(pred_norm.item()))  # [0, 5]
