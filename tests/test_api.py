"""
Tests de la API FastAPI.

Estrategia: usamos `TestClient` de FastAPI (visto en clase, slide 55 del tema 2)
para testear la API SIN levantar uvicorn ni cargar el modelo real.

¿Por qué mockear el modelo?
  - Cargar el .ckpt en CI tarda >30s y necesita el archivo en disco.
  - Los tests de API deben probar lógica HTTP (rutas, validación, errores),
    no la calidad del modelo.
  - La inferencia real está cubierta por test_pipeline.py.

Cómo se mockea: parcheamos `STSPredictor` antes de importar la app para que
NO cargue el modelo. Inyectamos un predictor falso que devuelve scores
deterministas.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest


# -----------------------------------------------------------------------------
# Fixture clave: TestClient con el predictor mockeado.
# Esto evita que la API real intente cargar el .ckpt.
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    # Si la app ya estaba importada (de otro test), la limpiamos
    for mod in ["src.inference_api"]:
        if mod in sys.modules:
            del sys.modules[mod]

    # Mockeamos STSPredictor antes de importar inference_api
    fake_predictor = MagicMock()
    fake_predictor.predict.return_value = 3.5  # score de prueba en [0, 5]

    with patch("src.inference.STSPredictor") as MockPredictor:
        MockPredictor.return_value = fake_predictor

        # Importamos AHORA (el lifespan usará el MockPredictor)
        from fastapi.testclient import TestClient
        from src.inference_api import app

        with TestClient(app) as c:
            # TestClient como context manager dispara el lifespan (carga modelo mock)
            yield c


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestRootEndpoint:
    """Tests del endpoint raíz GET /."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_json(self, client):
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "model" in data


class TestHealthEndpoint:
    """Tests del healthcheck GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_model_loaded(self, client):
        """Con el mock, el predictor existe en app.state, así que debe estar True."""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Tests del endpoint principal POST /predict."""

    def test_predict_valid_request(self, client):
        """Llamada bien formada devuelve 200 + score."""
        response = client.post("/predict", json={
            "sentence1": "A man is playing a guitar.",
            "sentence2": "A guy is playing the guitar.",
        })
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert "model" in data
        assert isinstance(data["score"], (int, float))

    def test_predict_score_in_valid_range(self, client):
        """El score debe estar en el rango [0, 5]."""
        response = client.post("/predict", json={
            "sentence1": "test", "sentence2": "test",
        })
        data = response.json()
        assert 0.0 <= data["score"] <= 5.0

    def test_predict_echoes_input(self, client):
        """La respuesta debe incluir las frases originales."""
        s1, s2 = "frase uno", "frase dos"
        response = client.post("/predict", json={"sentence1": s1, "sentence2": s2})
        data = response.json()
        assert data["sentence1"] == s1
        assert data["sentence2"] == s2

    def test_predict_missing_field(self, client):
        """Falta `sentence2` → Pydantic devuelve 422 (Unprocessable Entity)."""
        response = client.post("/predict", json={"sentence1": "hola"})
        assert response.status_code == 422

    def test_predict_empty_sentence(self, client):
        """Frase vacía → Pydantic devuelve 422 (min_length=1)."""
        response = client.post("/predict", json={"sentence1": "", "sentence2": "hola"})
        assert response.status_code == 422

    def test_predict_wrong_type(self, client):
        """Tipo incorrecto (int en vez de string) → 422."""
        response = client.post("/predict", json={"sentence1": 123, "sentence2": "hola"})
        assert response.status_code == 422

    def test_predict_get_not_allowed(self, client):
        """GET sobre /predict no está permitido (sólo POST)."""
        response = client.get("/predict")
        assert response.status_code == 405  # Method Not Allowed
