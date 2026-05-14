"""
Sube los 5 experimentos del notebook a W&B sin reentrenar.

Lee los CSVs de métricas + hparams.yaml de cada experimento y crea un run
en W&B por experimento. Cada run lleva:
  - Métricas época por época (replicadas con wandb.log).
  - Hiperparámetros (config) leídos de hparams.yaml + completados manualmente.
  - El .ckpt como artifact de W&B (versionado de modelos del temario).

Estructura esperada:
    checkpoints/
        1_Crossencoder/
            model.ckpt
            logs/version_0/
                metrics.csv
                hparams.yaml
        2_Biencoder/...
        ...

Uso:
    python -m scripts.relog_experiments --checkpoints-dir checkpoints/
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import wandb
import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Metadatos de cada experimento (del notebook original)
# Nota: SIN emojis en notes ni tags (W&B backend usa utf8mb3 en algunos campos)
# -----------------------------------------------------------------------------
EXPERIMENTS = {
    "1_Crossencoder": {
        "name": "exp1-crossencoder-roberta",
        "tags": ["cross-encoder", "roberta", "full-finetuning"],
        "config_extra": {
            "experiment_id": 1,
            "architecture": "cross-encoder",
            "model_base": "FacebookAI/roberta-base",
            "strategy": "full-finetuning",
            "loss": "MSELoss",
            "lora": False,
        },
        "notes": "Cross-encoder con RoBERTa-base. Fine-tuning con capas congeladas.",
    },
    "2_Biencoder": {
        "name": "exp2-biencoder-roberta",
        "tags": ["bi-encoder", "roberta", "full-finetuning"],
        "config_extra": {
            "experiment_id": 2,
            "architecture": "bi-encoder",
            "model_base": "FacebookAI/roberta-base",
            "strategy": "full-finetuning",
            "loss": "CosineSimilarityLoss",
            "lora": False,
        },
        "notes": "Bi-encoder con RoBERTa-base. Fine-tuning con capas congeladas.",
    },
    "3_Biencoder_minilm": {
        "name": "exp3-biencoder-minilm",
        "tags": ["bi-encoder", "minilm", "full-finetuning"],
        "config_extra": {
            "experiment_id": 3,
            "architecture": "bi-encoder",
            "model_base": "sentence-transformers/all-MiniLM-L6-v2",
            "strategy": "full-finetuning",
            "loss": "CosineSimilarityLoss",
            "lora": False,
        },
        "notes": "Bi-encoder con MiniLM-L6-v2. Fine-tuning con capas congeladas.",
    },
    "4_lora": {
        "name": "exp4-crossencoder-lora",
        "tags": ["cross-encoder", "roberta", "lora", "winner"],
        "config_extra": {
            "experiment_id": 4,
            "architecture": "cross-encoder",
            "model_base": "FacebookAI/roberta-base",
            "strategy": "lora",
            "loss": "MSELoss",
            "lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_target_modules": "query,value",
        },
        "notes": "GANADOR. Cross-encoder con RoBERTa + LoRA. Mejor Pearson de los 5 experimentos.",
    },
    "5_lora_minilm": {
        "name": "exp5-biencoder-minilm-lora",
        "tags": ["bi-encoder", "minilm", "lora"],
        "config_extra": {
            "experiment_id": 5,
            "architecture": "bi-encoder",
            "model_base": "sentence-transformers/all-MiniLM-L6-v2",
            "strategy": "lora",
            "loss": "CosineSimilarityLoss",
            "lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_target_modules": "query,value",
        },
        "notes": "Bi-encoder con MiniLM + LoRA.",
    },
}


# -----------------------------------------------------------------------------
# Funciones auxiliares
# -----------------------------------------------------------------------------
def load_hparams(hparams_path: Path) -> dict:
    """Lee hparams.yaml de Lightning si existe."""
    if not hparams_path.exists():
        logger.warning(f"  No se encontró {hparams_path}, sigo sin él.")
        return {}
    with open(hparams_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """Lee metrics.csv del CSVLogger de Lightning."""
    return pd.read_csv(csv_path)


def find_checkpoint(exp_dir: Path) -> Optional[Path]:
    """Encuentra el .ckpt o .ckp dentro del directorio del experimento."""
    candidates = list(exp_dir.glob("*.ckpt")) + list(exp_dir.glob("*.ckp"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def log_metrics_to_wandb(df: pd.DataFrame, run) -> None:
    """Reenvía las métricas del CSV a W&B como si fueran logs en vivo."""
    if "step" in df.columns:
        df = df.sort_values("step")

    n_logged = 0
    for _, row in df.iterrows():
        log_dict = {}
        for col in df.columns:
            if col == "step":
                continue
            val = row[col]
            if pd.notna(val):
                log_dict[col] = float(val) if not isinstance(val, str) else val

        if not log_dict:
            continue

        step = int(row["step"]) if "step" in row and pd.notna(row["step"]) else None
        run.log(log_dict, step=step)
        n_logged += 1

    logger.info(f"  {n_logged} filas de métricas enviadas a W&B")


def get_summary_from_metrics(df: pd.DataFrame) -> dict:
    """Extrae las métricas finales (best val_pearson y test_pearson) para el summary."""
    summary = {}
    if "val_pearson" in df.columns:
        best_val = df["val_pearson"].max()
        if pd.notna(best_val):
            summary["best_val_pearson"] = float(best_val)
    if "test_pearson" in df.columns:
        test_vals = df["test_pearson"].dropna()
        if len(test_vals) > 0:
            summary["final_test_pearson"] = float(test_vals.iloc[-1])
    if "test_loss" in df.columns:
        test_loss = df["test_loss"].dropna()
        if len(test_loss) > 0:
            summary["final_test_loss"] = float(test_loss.iloc[-1])
    return summary


def sanitize_for_wandb(obj):
    """
    Elimina caracteres 4-byte UTF-8 (emojis y similares) que el backend
    de W&B (utf8mb3 en algunas columnas) no acepta. Lo aplicamos a config
    y al resto de strings que vamos a enviar.
    """
    if isinstance(obj, dict):
        return {sanitize_for_wandb(k): sanitize_for_wandb(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_wandb(x) for x in obj]
    if isinstance(obj, str):
        # Eliminar caracteres BMP por encima del rango básico (4-byte UTF-8)
        return "".join(c for c in obj if ord(c) < 0x10000 and not (0xD800 <= ord(c) <= 0xDFFF))
    return obj


# -----------------------------------------------------------------------------
# Procesado de un experimento
# -----------------------------------------------------------------------------
def relog_experiment(
    exp_dir: Path,
    exp_meta: dict,
    project: str,
    upload_ckpt: bool = True,
) -> None:
    """Sube un experimento completo a W&B."""
    logger.info("=" * 70)
    logger.info(f"Procesando: {exp_dir.name} -> {exp_meta['name']}")

    # 1. Localizar CSV y hparams
    log_dir = exp_dir / "logs" / "version_0"
    csv_path = log_dir / "metrics.csv"
    hparams_path = log_dir / "hparams.yaml"

    if not csv_path.exists():
        logger.error(f"  No existe {csv_path}, saltando experimento.")
        return

    # 2. Cargar hparams + config_extra (sanitizados)
    hparams = load_hparams(hparams_path)
    config = sanitize_for_wandb({**hparams, **exp_meta["config_extra"]})

    # 3. Crear el run en W&B (sin reinit deprecado; usamos finish_previous)
    run = wandb.init(
        project=project,
        name=sanitize_for_wandb(exp_meta["name"]),
        tags=[sanitize_for_wandb(t) for t in exp_meta["tags"]],
        notes=sanitize_for_wandb(exp_meta["notes"]),
        config=config,
        reinit="finish_previous",
    )

    # 4. Replicar las métricas
    df = load_metrics(csv_path)
    logger.info(f"  Filas en CSV: {len(df)}, columnas: {list(df.columns)}")
    log_metrics_to_wandb(df, run)

    # 5. Establecer summary con métricas finales
    summary = get_summary_from_metrics(df)
    for k, v in summary.items():
        run.summary[k] = v
    if summary:
        logger.info(f"  Summary: {summary}")

    # 6. Subir el .ckpt como artifact
    if upload_ckpt:
        ckpt_path = find_checkpoint(exp_dir)
        if ckpt_path is not None:
            size_mb = ckpt_path.stat().st_size / 1e6
            logger.info(f"  Subiendo checkpoint: {ckpt_path.name} ({size_mb:.1f} MB)")
            artifact = wandb.Artifact(
                name=exp_meta["name"] + "-model",
                type="model",
                description=f"Checkpoint final del {exp_meta['name']}",
                metadata=sanitize_for_wandb(summary),
            )
            artifact.add_file(str(ckpt_path))
            run.log_artifact(artifact)
            logger.info("  Artifact subido")
        else:
            logger.warning(f"  No se encontró checkpoint en {exp_dir}")

    run.finish()
    logger.info(f"  Run completado: {exp_meta['name']}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints-dir", type=str, required=True,
        help="Carpeta con los 5 subdirectorios de experimentos",
    )
    parser.add_argument(
        "--project", type=str, default="sts-mlops",
        help="Nombre del proyecto en W&B",
    )
    parser.add_argument(
        "--no-upload-ckpt", action="store_true",
        help="No subir los .ckpt como artifacts (acelera y ahorra storage)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Procesar solo un experimento (nombre de la carpeta, ej: '4_lora')",
    )
    args = parser.parse_args()

    base = Path(args.checkpoints_dir)
    if not base.exists():
        raise FileNotFoundError(f"No existe la carpeta: {base}")

    for folder_name, meta in EXPERIMENTS.items():
        if args.only and folder_name != args.only:
            continue
        exp_dir = base / folder_name
        if not exp_dir.exists():
            logger.warning(f"No existe {exp_dir}, saltando.")
            continue
        relog_experiment(
            exp_dir=exp_dir,
            exp_meta=meta,
            project=args.project,
            upload_ckpt=not args.no_upload_ckpt,
        )

    logger.info("=" * 70)
    logger.info("Todos los experimentos procesados.")
    logger.info(f"   Ve a tu W&B -> proyecto '{args.project}' para verlos.")


if __name__ == "__main__":
    main()