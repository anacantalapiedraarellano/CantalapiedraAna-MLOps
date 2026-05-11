"""
Test de pipeline: verifica que el pipeline de entrenamiento corre end-to-end
sin reventar, usando un mini-batch y muy pocas iteraciones.

NO mockeamos nada aquí: queremos asegurar que el modelo, los datos y el
optimizador encajan correctamente. Este test es más lento (~2-3 min en CPU)
pero es el que da confianza real de que el entrenamiento no está roto.

Estrategia para que sea rápido:
  - max_epochs=1
  - limit_train_batches=2 (sólo 2 batches de train)
  - limit_val_batches=2 (sólo 2 batches de val)
  - batch_size=4 (mini-batch pequeño)
  - Sin W&B, sin checkpoints persistentes
"""
import tempfile
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

from src.config import MODEL_NAME, OPTIMIZER_PARAMS, POOLING, get_lora_config
from src.data import STSDataModule
from src.model import STSRegressorLoRA


@pytest.mark.slow
def test_full_training_pipeline_runs():
    """
    Verifica que entrenamiento + validación + test corren sin errores
    durante 1 mini-época con muy pocos batches.
    """
    pl.seed_everything(42, workers=True)

    # DataModule con batch pequeño
    data_module = STSDataModule(
        tokenizer_name=MODEL_NAME,
        max_length=64,   # más corto para acelerar
        batch_size=4,
        num_workers=0,
    )

    # Modelo (mismo del Exp 4, sólo cambian iteraciones)
    model = STSRegressorLoRA(
        model_name=MODEL_NAME,
        lora_config=get_lora_config(),
        optimizer_params=OPTIMIZER_PARAMS,
        pooling=POOLING,
    )

    # Trainer minimal en directorio temporal
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=2,
            limit_test_batches=2,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=tmp_dir,
        )

        # Entrenamiento
        trainer.fit(model, data_module)

        # Test
        results = trainer.test(model, data_module, verbose=False)

    # Comprobaciones
    assert len(results) == 1
    assert "test_loss" in results[0]
    assert "test_pearson" in results[0]

    # La loss y el pearson deben ser números finitos
    assert torch.isfinite(torch.tensor(results[0]["test_loss"]))


def test_model_forward_pass():
    """Test rápido: el forward pass del modelo no revienta y devuelve scores en [0, 1]."""
    pl.seed_everything(42)

    model = STSRegressorLoRA(
        model_name=MODEL_NAME,
        lora_config=get_lora_config(),
        optimizer_params=OPTIMIZER_PARAMS,
        pooling=POOLING,
    )
    model.eval()

    # Input dummy con shape esperada
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    # Output debe ser (batch_size,) y en [0, 1] (la sigmoide del final)
    assert output.shape == (batch_size,)
    assert (output >= 0).all() and (output <= 1).all()
