# STS-MLOps: Similitud Textual Semántica con Cross-encoder + LoRA

Proyecto final de la asignatura **MLOps** del Máster en Deep Learning (UPM).

**Autor:** [TU NOMBRE Y APELLIDOS]

## Descripción

Sistema de estimación de similitud semántica entre pares de frases (regresión continua en escala 0-5)
basado en un **Cross-encoder con RoBERTa + LoRA** entrenado sobre el STS Benchmark.

Resultado: **Pearson 0.8671 en test**.

## Estructura del proyecto

```
.
├── src/                # Código de producción
│   ├── config.py       # Hiperparámetros centralizados
│   ├── data.py         # DataModule del STS Benchmark
│   ├── model.py        # Cross-encoder + LoRA (PyTorch Lightning)
│   ├── train.py        # Script de entrenamiento con W&B
│   └── inference_api.py  # API FastAPI (próximamente)
├── tests/              # Tests unitarios, API y pipeline
├── notebooks/          # Notebook exploratorio original
├── models/             # Checkpoints (no versionados)
├── requirements.txt
└── README.md
```

## Instalación local

```bash
python -m venv venv
source venv/bin/activate           # Linux/Mac
# venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

## Entrenamiento

```bash
# Con W&B (requiere `wandb login` previo)
python -m src.train

# Sin W&B (debug local)
python -m src.train --no-wandb --epochs 2
```

## Próximos pasos (en construcción)

- [ ] API de inferencia (FastAPI)
- [ ] Tests
- [ ] Dockerfile
- [ ] CI/CD con GitHub Actions
- [ ] Despliegue en producción
- [ ] W&B Report
