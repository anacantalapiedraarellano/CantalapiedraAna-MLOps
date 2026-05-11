"""
Configuración compartida de pytest.

- Añade el directorio raíz al sys.path para que `from src.xxx import yyy` funcione.
- Define la marca `slow` para tests lentos (test_pipeline).
"""
import sys
from pathlib import Path


# Añadir la raíz del proyecto al PYTHONPATH para que src/ sea importable
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """Registra la marca 'slow' para que pytest no se queje."""
    config.addinivalue_line(
        "markers", "slow: tests lentos (entrenamiento, carga de modelo real)"
    )
