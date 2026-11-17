"""Generate the calibrated PDF attachments of the stress test (block 3).

Writes ``attach_{5,20,50,100}kb.pdf`` next to this file. The size refers to
the EXTRACTED TEXT (what pypdf hands to the prompt), not to the bytes on disk:
the runner cares about the prompt budget. ``0 KB`` is the absence of an
attachment, so there is no file for it.

Each document carries two unique markers so the run can measure what the
model actually saw:

- ``HEAD_FACT`` in the first paragraph — survives any truncation.
- ``TAIL_FACT`` in the last paragraph — lost once the text exceeds
  ``MAX_ATTACHMENT_CHARS`` (20.000 chars in this service), because the
  extractor keeps only the beginning.

Deterministic: same paragraph, same repetitions, same bytes on every run. The
PDFs are gitignored; commit only this script and rebuild them with::

    uv run python -m evals.stress.fixtures.build_pdfs
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from fpdf import FPDF
from pypdf import PdfReader

FIXTURES_DIR = Path(__file__).parent
SIZES_KB = (5, 20, 50, 100)

# Single words on purpose: fpdf2 wraps lines and pypdf returns "Thales\nLuna".
HEAD_FACT = "Thales"
TAIL_FACT = "Solace"

_HEAD = (
    "ANEXO TECNICO DEL CLIENTE. Requisito obligatorio: todas las claves "
    f"criptograficas se custodian en un HSM {HEAD_FACT} Luna propiedad del cliente.\n"
)
_TAIL = (
    "\nRequisito final del anexo: la mensajeria entre sistemas usa el bus "
    f"{TAIL_FACT} PubSub+ ya licenciado por el cliente.\n"
)
# NOTA: relleno neutro a proposito. No menciona ninguno de los hechos que
# siguen los escenarios (nombres, tecnologias, cifras): si los mencionara, el
# adjunto "recordaria" hechos y falsearia la metrica de memoria.
_PARAGRAPH = (
    "El proveedor documentara cada entrega con notas de version, manual de "
    "usuario y guia de despliegue. Las reuniones de seguimiento seran "
    "quincenales y quedara acta firmada por ambas partes. Se mantendran tres "
    "entornos separados: desarrollo, preproduccion y produccion, con datos "
    "anonimizados fuera de produccion. Toda incidencia critica tendra respuesta "
    "en menos de cuatro horas laborables. El codigo pasara revision por pares y "
    "un analisis estatico antes de cada fusion, y la cobertura de pruebas "
    "automaticas no bajara del setenta por ciento. La aceptacion de cada hito "
    "requiere una demostracion funcional y la validacion escrita del "
    "responsable de negocio. "
)


def build_pdf(target_kb: int, path: Path) -> tuple[int, int]:
    """Write a PDF whose extracted text is ~``target_kb`` KB. Returns (chars, bytes)."""
    target_chars = target_kb * 1024
    body: list[str] = [_HEAD]
    chars = len(_HEAD) + len(_TAIL)
    while chars < target_chars:
        body.append(_PARAGRAPH)
        chars += len(_PARAGRAPH)
    body.append(_TAIL)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    # NOTA: fijar la fecha de creacion hace el fichero byte a byte reproducible.
    pdf.set_creation_date(datetime(2026, 1, 1, tzinfo=UTC))
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for block in body:
        pdf.multi_cell(0, 5, block)
        pdf.ln(2)
    pdf.output(str(path))
    return chars, path.stat().st_size


def extracted_chars(path: Path) -> int:
    """Length of the text pypdf extracts, the same way app/services/attachments.py does."""
    reader = PdfReader(str(path))
    return len("\n\n".join((p.extract_text() or "").strip() for p in reader.pages).strip())


def fixture_path(size_kb: int) -> Path:
    return FIXTURES_DIR / f"attach_{size_kb}kb.pdf"


def main() -> int:
    for kb in SIZES_KB:
        path = fixture_path(kb)
        chars, size = build_pdf(kb, path)
        print(f"{path.name}: text={chars} chars, file={size} bytes, pypdf={extracted_chars(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
