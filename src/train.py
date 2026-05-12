"""
Script de entrenamiento profesional del Experimento 4 (Cross-encoder + LoRA).

Características:
  - Reproducibilidad (seed fijo).
  - Hiperparámetros parametrizables vía CLI.
  - Logging dual: W&B (métricas online) + logging estándar (consola).
  - Checkpoint del mejor modelo según val_pearson.
  - EarlyStopping para evitar overfitting.

Uso:
    python -m src.train --epochs 50 --batch-size 32 --lr 2e-5
    python -m src.train --no-wandb   # entrenamiento sin W&B (debug local)
"""
import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from src.config import (
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    POOLING,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    OPTIMIZER_PARAMS,
    SEED,
    MODELS_DIR,
    LOGS_DIR,
    WANDB_PROJECT,
    WANDB_ENTITY,
    get_lora_config,
)
from src.data import STSDataModule
from src.model import STSRegressorLoRA


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    """Fija las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento del Cross-encoder LoRA para STS")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--lr", type=float, default=OPTIMIZER_PARAMS["lr"])
    parser.add_argument("--weight-decay", type=float, default=OPTIMIZER_PARAMS["weight_decay"])
    parser.add_argument("--pooling", type=str, default=POOLING, choices=["mean", "cls"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--models-dir", type=str, default=MODELS_DIR)
    parser.add_argument("--logs-dir", type=str, default=LOGS_DIR)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Desactiva W&B (útil para debug local)")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Nombre del run en W&B (por defecto autogenerado)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Crear directorios
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Modelo base: {args.model_name}")

    # Hiperparámetros para tracking en W&B
    optimizer_params = {"lr": args.lr, "weight_decay": args.weight_decay}
    hparams = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "pooling": args.pooling,
        "seed": args.seed,
        "experiment": "exp4_crossencoder_lora",
        **optimizer_params,
        # LoRA hparams
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "lora_target_modules": "query,value",
    }

    # DataModule
    logger.info("Cargando dataset STS Benchmark...")
    data_module = STSDataModule(
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    data_module.setup()

    # Modelo
    logger.info("Inicializando modelo Cross-encoder + LoRA...")
    model = STSRegressorLoRA(
        model_name=args.model_name,
        lora_config=get_lora_config(),
        optimizer_params=optimizer_params,
        pooling=args.pooling,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parámetros entrenables: {n_trainable:,}")

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.models_dir,
        monitor="val_pearson",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_pearson:.4f}",
    )
    early_stop_cb = EarlyStopping(
        monitor="val_pearson",
        mode="max",
        patience=args.patience,
    )

    # Logger: W&B en producción, CSV como fallback
    if args.no_wandb:
        pl_logger = CSVLogger(save_dir=args.logs_dir, name="exp4_crossencoder_lora")
        logger.info("W&B desactivado, usando CSVLogger.")
    else:
        pl_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=hparams,
            log_model=True,  # sube el checkpoint como artifact al final
        )
        logger.info(f"W&B activado. Project: {args.wandb_project}")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=pl_logger,
        deterministic=False,  # True ralentiza mucho, el seed ya se fija aparte
    )

    # Entrenamiento
    logger.info("Iniciando entrenamiento...")
    trainer.fit(model, data_module)
    best_val_pearson = checkpoint_cb.best_model_score.item()
    logger.info(f"Mejor val_pearson: {best_val_pearson:.4f}")
    logger.info(f"Checkpoint guardado: {checkpoint_cb.best_model_path}")

    # Evaluación final en test
    logger.info("Evaluando en test...")
    results = trainer.test(ckpt_path="best", datamodule=data_module)
    test_pearson = results[0]["test_pearson"]
    logger.info(f"Test Pearson: {test_pearson:.4f}")

    # Cerramos W&B
    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
