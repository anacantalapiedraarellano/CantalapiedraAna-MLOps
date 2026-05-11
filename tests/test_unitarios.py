"""
Tests unitarios: prueban funciones aisladas sin levantar la API ni cargar
el modelo real. Son los tests más rápidos y baratos del proyecto.

Se centran en:
  - normalize_score / denormalize_score: la lógica de conversión [0,5] <-> [0,1].
  - STSCollateFn: que tokeniza correctamente pares de frases.
  - get_lora_config: que la configuración LoRA es la esperada.
"""
import pytest
import torch

from src.data import normalize_score, denormalize_score, STSCollateFn
from src.config import MODEL_NAME, MAX_LENGTH, get_lora_config


# -----------------------------------------------------------------------------
# normalize_score / denormalize_score
# -----------------------------------------------------------------------------
class TestNormalization:
    """Tests de las funciones de normalización de scores."""

    def test_normalize_zero(self):
        assert normalize_score(0.0) == 0.0

    def test_normalize_max(self):
        assert normalize_score(5.0) == 1.0

    def test_normalize_middle(self):
        assert normalize_score(2.5) == 0.5

    def test_denormalize_zero(self):
        assert denormalize_score(0.0) == 0.0

    def test_denormalize_max(self):
        assert denormalize_score(1.0) == 5.0

    def test_normalize_denormalize_is_identity(self):
        """denormalize(normalize(x)) == x para varios valores."""
        for score in [0.0, 1.5, 2.5, 3.7, 5.0]:
            assert denormalize_score(normalize_score(score)) == pytest.approx(score)


# -----------------------------------------------------------------------------
# STSCollateFn
# -----------------------------------------------------------------------------
class TestCollateFn:
    """Tests del collate function que tokeniza pares de frases."""

    @pytest.fixture(scope="class")
    def collate(self):
        # scope="class": el tokenizer se instancia una vez para toda la clase
        return STSCollateFn(tokenizer_name=MODEL_NAME, max_length=MAX_LENGTH)

    def test_collate_returns_correct_types(self, collate):
        batch = [
            {"sentence1": "Hello world", "sentence2": "Hi world", "score": 4.0},
            {"sentence1": "The cat", "sentence2": "The dog", "score": 2.0},
        ]
        encoded, labels = collate(batch)
        assert isinstance(encoded["input_ids"], torch.Tensor)
        assert isinstance(encoded["attention_mask"], torch.Tensor)
        assert isinstance(labels, torch.Tensor)

    def test_collate_batch_size(self, collate):
        """Los tensors de salida deben tener tantas filas como ejemplos."""
        batch = [
            {"sentence1": "A", "sentence2": "B", "score": 1.0},
            {"sentence1": "C", "sentence2": "D", "score": 2.0},
            {"sentence1": "E", "sentence2": "F", "score": 3.0},
        ]
        encoded, labels = collate(batch)
        assert encoded["input_ids"].shape[0] == 3
        assert labels.shape == (3,)

    def test_collate_labels_normalized(self, collate):
        """Los labels deben estar normalizados a [0, 1]."""
        batch = [
            {"sentence1": "A", "sentence2": "B", "score": 0.0},
            {"sentence1": "C", "sentence2": "D", "score": 5.0},
            {"sentence1": "E", "sentence2": "F", "score": 2.5},
        ]
        _, labels = collate(batch)
        assert labels[0].item() == pytest.approx(0.0)
        assert labels[1].item() == pytest.approx(1.0)
        assert labels[2].item() == pytest.approx(0.5)

    def test_collate_respects_max_length(self, collate):
        """Frases muy largas deben truncarse al max_length configurado."""
        long_sentence = " ".join(["word"] * 500)
        batch = [{"sentence1": long_sentence, "sentence2": long_sentence, "score": 3.0}]
        encoded, _ = collate(batch)
        assert encoded["input_ids"].shape[1] <= MAX_LENGTH


# -----------------------------------------------------------------------------
# LoRA config
# -----------------------------------------------------------------------------
class TestLoraConfig:
    """Tests de la configuración LoRA del Exp 4."""

    def test_lora_config_returns_object(self):
        config = get_lora_config()
        assert config is not None

    def test_lora_config_target_modules(self):
        """LoRA debe aplicarse a query y value (Exp 4)."""
        config = get_lora_config()
        assert "query" in config.target_modules
        assert "value" in config.target_modules

    def test_lora_config_hyperparams(self):
        """Hiperparámetros LoRA del Exp 4: r=8, alpha=16, dropout=0.1."""
        config = get_lora_config()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
