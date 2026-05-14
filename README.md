# STS-MLOps: Similitud Textual Semántica con Cross-encoder + LoRA

Proyecto final de la asignatura **MLOps** del Máster en Deep Learning de la
Universidad Politécnica de Madrid.

**Autora:** Ana Cantalapiedra Arellano

---

## Descripción

Sistema de **Similitud Textual Semántica (STS)**: dado un par de frases en inglés,
estima un score continuo de similitud en el rango **0 (sin relación) – 5 (idénticas)**.

El modelo en producción es un **Cross-encoder basado en RoBERTa-base afinado con LoRA**,
seleccionado tras comparar cinco configuraciones distintas (cross-encoder vs bi-encoder,
RoBERTa vs MiniLM, fine-tuning vs LoRA) sobre el dataset **STS Benchmark**. Es el que
obtiene mejor correlación de Pearson en el conjunto de test.

Este repositorio aplica las metodologías y herramientas de MLOps vistas en la asignatura:
estructura de proyecto reproducible, código modular, API de inferencia, tests,
contenedorización con Docker, integración continua con GitHub Actions, seguimiento de
experimentos con Weights & Biases y despliegue en producción.

## Nota sobre el entrenamiento y los modelos

Los cinco modelos comparados en este proyecto **fueron entrenados previamente** en la
asignatura de NLP del máster, y sus checkpoints y métricas se reutilizan aquí. Esta
decisión es deliberada: el objetivo del proyecto de MLOps es **industrializar un modelo
ya existente** (estructura, API, tests, contenedor, CI/CD, despliegue y seguimiento de
experimentos), no volver a entrenarlo.

Además, el reentrenamiento completo era **inviable con los recursos de cómputo
disponibles**: en una prueba de validación del pipeline, una sola época de entrenamiento
en CPU superaba la hora de ejecución. Por ello:

- El pipeline de entrenamiento (`src/train.py`) es **completamente funcional** y se ha
  verificado ejecutándolo de forma reducida, pero el modelo servido en producción es el
  checkpoint del experimento ganador ya entrenado.
- Las métricas de los cinco experimentos se han subido a Weights & Biases mediante el
  script `scripts/relog_experiments.py`, que registra en W&B los resultados (métricas
  época a época, hiperparámetros y checkpoints) **sin necesidad de reentrenar**.

Los checkpoints de los cinco experimentos y el modelo en producción están
disponibles para descarga (ver sección *Configuración del entorno local*).

## Enlaces del proyecto

- **Repositorio GitHub:** https://github.com/anacantalapiedraarellano/CantalapiedraAna-MLOps
- **Weights & Biases (proyecto):** https://wandb.ai/anacantalapiedra-lab/sts-mlops
- **Weights & Biases (Report):** https://wandb.ai/anacantalapiedra-lab/sts-mlops/reports/An-lisis-comparativo-Similitud-Textual-Sem-ntica-con-Cross-encoder-LoRA--VmlldzoxNjg3ODk0Nw
- **Endpoint en producción (Hugging Face Spaces):** https://huggingface.co/spaces/anacantalapiedraarellano/sts-mlops
- **Documentación interactiva de la API:** https://anacantalapiedraarellano-sts-mlops.hf.space/docs

## Estructura del proyecto

```
.
├── src/                      # Código de producción
│   ├── config.py             # Hiperparámetros centralizados
│   ├── data.py               # DataModule del STS Benchmark + normalización
│   ├── model.py              # Cross-encoder + LoRA (PyTorch Lightning)
│   ├── train.py              # Script de entrenamiento con integración W&B
│   ├── inference.py          # Lógica de inferencia (clase STSPredictor)
│   └── inference_api.py      # API REST con FastAPI
├── tests/                    # Tests automatizados
│   ├── test_unitarios.py     # Tests de funciones de preprocesado y config
│   ├── test_api.py           # Tests de la API con TestClient (modelo mockeado)
│   └── test_pipeline.py      # Test de entrenamiento end-to-end (mini-batch)
├── scripts/
│   └── relog_experiments.py  # Sube los experimentos del notebook a W&B
├── notebooks/
│   └── NLP_STS.ipynb         # Notebook exploratorio original
├── models/                   # Checkpoint del modelo (no versionado en Git)
├── .github/workflows/ci.yml  # Pipeline de integración continua
├── Dockerfile                # Imagen de la API para producción
├── .dockerignore
├── requirements.txt          # Dependencias con versiones fijadas
└── README.md
```

## Funcionalidades principales

- **Entrenamiento parametrizable** (`src/train.py`): entrena el cross-encoder con LoRA,
  con hiperparámetros configurables por línea de comandos, seguimiento en W&B,
  checkpoint del mejor modelo y early stopping.
- **API de inferencia** (`src/inference_api.py`): API REST con FastAPI que sirve el
  modelo, con validación de entrada (Pydantic), carga del modelo en arranque,
  manejo de errores y logging.
- **Tests automatizados** (`tests/`): tests unitarios, de API y de pipeline,
  ejecutados automáticamente en cada push mediante GitHub Actions.
- **Contenedorización** (`Dockerfile`): la API se empaqueta en una imagen Docker
  lista para desplegar.
- **Seguimiento de experimentos** (W&B): los cinco experimentos del proyecto están
  registrados en Weights & Biases con sus métricas, hiperparámetros y checkpoints.

## Endpoints de la API

| Método | Ruta | Descripción |
|--------|----------|------------------------------------------|
| GET    | `/`        | Información general de la API.          |
| GET    | `/health`  | Healthcheck (estado y modelo cargado).  |
| POST   | `/predict` | Predice la similitud entre dos frases.  |
| GET    | `/docs`    | Documentación interactiva (Swagger).    |

### Ejemplo de petición

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sentence1": "A man is playing a guitar", "sentence2": "A guy is playing the guitar"}'
```

Respuesta:

```json
{
  "score": 4.52,
  "model": "exp4-crossencoder-lora",
  "sentence1": "A man is playing a guitar",
  "sentence2": "A guy is playing the guitar"
}
```

---

## Configuración del entorno local de desarrollo

### Requisitos previos

- Python 3.10
- pip
- (Opcional) Docker, para ejecutar la API en contenedor

### 1. Clonar el repositorio

```bash
git clone https://github.com/anacantalapiedraarellano/CantalapiedraAna-MLOps.git
cd CantalapiedraAna-MLOps
```

### 2. Crear el entorno virtual e instalar dependencias

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Colocar el checkpoint del modelo

Los checkpoints de los modelos no se versionan en Git ni se incluyen en el
entregable comprimido por su tamaño (varios cientos de MB cada uno). Están
disponibles para descarga en los siguientes enlaces:

- **Modelo en producción** (`best_model.ckpt`, Experimento 4 - Cross-encoder + LoRA):
  [DESCARGAR AQUÍ](https://drive.google.com/file/d/1evBOR9s8RZyvkmVodi5wL6Q72tqQZlpn/view?usp=drive_link)
- **Todos los experimentos** (carpeta `checkpoints/` con los 5 modelos, sus métricas
  y logs, usada por el script de relogging a W&B):
  [DESCARGAR AQUÍ](https://drive.google.com/drive/folders/130PImhqoHgJt8VHgFtt6MPLF_AT3LaBK?usp=drive_link)

Para ejecutar la API en local, descarga `best_model.ckpt` y colócalo en la carpeta
`models/`:

​```
models/best_model.ckpt
​```
> Nota: la API en producción (Hugging Face Spaces) ya incluye el modelo embebido,
> por lo que puede probarse directamente sin necesidad de descargar nada.

---

## Uso

### Entrenar el modelo

```bash
# Entrenamiento con seguimiento en W&B (requiere `wandb login` previo)
python -m src.train

# Entrenamiento sin W&B (para pruebas locales rápidas)
python -m src.train --no-wandb --epochs 2
```

Parámetros configurables: `--epochs`, `--batch-size`, `--lr`, `--patience`, etc.
Ver `python -m src.train --help` para la lista completa.

> **Nota:** el entrenamiento completo requiere GPU. En CPU, una sola época supera
> la hora de ejecución, por lo que se recomienda usar el checkpoint ya entrenado
> incluido en el proyecto (ver sección *Nota sobre el entrenamiento y los modelos*).

### Lanzar la API en local

```bash
uvicorn src.inference_api:app --host 0.0.0.0 --port 8000
```

La API quedará disponible en `http://localhost:8000`. La documentación interactiva
está en `http://localhost:8000/docs`.

### Ejecutar los tests

```bash
# Tests rápidos (unitarios + API)
pytest -v -m "not slow"

# Todos los tests, incluido el de pipeline (entrena un mini-batch)
pytest -v
```

### Ejecutar con Docker

```bash
# Construir la imagen
docker build -t sts-mlops:latest .

# Ejecutar el contenedor
docker run -p 8000:8000 sts-mlops:latest
```

---

## Modelo

| | |
|------------------------|----------------------------------------------|
| Arquitectura           | Cross-encoder                                |
| Modelo base            | FacebookAI/roberta-base                      |
| Técnica de adaptación  | LoRA (r=8, alpha=16, dropout=0.1)            |
| Módulos LoRA           | query, value                                 |
| Dataset                | STS Benchmark (mteb/stsbenchmark-sts)        |
| Función de pérdida     | MSELoss                                      |
| Optimizador            | AdamW (lr=2e-5, weight_decay=0.01)           |

El análisis comparativo completo de los cinco experimentos está disponible en el
[W&B Report](https://wandb.ai/anacantalapiedra-lab/sts-mlops/reports/An-lisis-comparativo-Similitud-Textual-Sem-ntica-con-Cross-encoder-LoRA--VmlldzoxNjg3ODk0Nw).

---

## Stack tecnológico

PyTorch · PyTorch Lightning · Hugging Face Transformers · PEFT (LoRA) ·
FastAPI · Docker · GitHub Actions · Weights & Biases · pytest