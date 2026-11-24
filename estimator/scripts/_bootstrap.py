"""Make ``app`` importable when running ``python scripts/<name>.py``.

NOTA: al ejecutar un script, Python añade a sys.path la carpeta del script
(scripts/), no la raíz del proyecto, y entonces ``import app`` falla. Este
módulo añade la raíz (estimator/ en local, /app en el contenedor) para que los
scripts funcionen igual dentro y fuera de Docker.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SAMPLE = ROOT / "data" / "budgets_sample.json"
