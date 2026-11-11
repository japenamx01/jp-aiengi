"""Local text extraction for PDF and Word attachments (path B of the exercise).

The text is extracted inside the service and rendered into the user message,
instead of uploading the file to a provider's Files API. That keeps every call
provider-agnostic — the Router can still fall back from OpenAI to Anthropic, and
the exact-match cache still works because the document text is part of the
prompt — and it is the first step towards chunking documents for RAG.

NOTA: los dos caminos del ejercicio
-----------------------------------
  A) Multimodal directo: se sube el PDF a la Files API de OpenAI/Anthropic y el
     mensaje lo referencia por id. Menos código, pero el archivo existe solo en
     ESE proveedor: el fallback a otro modelo dejaría de verlo.
  B) Extracción local (este módulo): pypdf / python-docx sacan el texto aquí y
     viaja como texto normal dentro del prompt. Más código, pero cualquier
     modelo lo entiende. Límite: un PDF escaneado (solo imágenes) no tiene texto
     que extraer; eso requeriría OCR o el camino A.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePath

import structlog
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

log = structlog.get_logger()

MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
# NOTA: el texto extraído entra en el prompt y además se queda en el historial
# de los turnos siguientes. Recortarlo protege la ventana de contexto y el coste.
MAX_ATTACHMENT_CHARS = 20_000
TRUNCATION_MARK = "\n[... attachment truncated ...]"
SUPPORTED_EXTENSIONS = (".pdf", ".docx")


class UnsupportedAttachmentError(ValueError):
    """The file type is not PDF or Word (.docx). Mapped to HTTP 415."""


class UnreadableAttachmentError(ValueError):
    """The file is too large, corrupt or has no extractable text. Mapped to HTTP 422."""


@dataclass(frozen=True)
class ExtractedAttachment:
    """Text pulled out of one uploaded file, ready to render into the prompt."""

    filename: str
    text: str
    truncated: bool

    @property
    def chars(self) -> int:
        return len(self.text)


def extract_attachment(filename: str, data: bytes) -> ExtractedAttachment:
    """Extract the text of a PDF or .docx file, truncated to ``MAX_ATTACHMENT_CHARS``."""
    # NOTA: se decide por la extensión. El content-type que manda el navegador
    # no es fiable (a veces llega "application/octet-stream"), y si alguien
    # renombra un .txt a .pdf, pypdf fallará y devolveremos 422.
    name = PurePath(filename or "attachment").name
    extension = PurePath(name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedAttachmentError(
            f"'{name}' is not supported. Attach PDF or Word (.docx) files."
        )
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise UnreadableAttachmentError(
            f"'{name}' exceeds the {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB limit."
        )

    try:
        text = _pdf_text(data) if extension == ".pdf" else _docx_text(data)
    except (
        PdfReadError,
        zipfile.BadZipFile,
        PackageNotFoundError,
        ValueError,
        KeyError,
        OSError,
    ) as exc:
        # NOTA: un PDF corrupto o un .docx que en realidad no es un zip (un
        # .docx es un zip con XML dentro) lanzan errores distintos según la
        # librería; todos acaban en el mismo 422.
        raise UnreadableAttachmentError(f"Could not read '{name}': {exc}") from exc

    text = text.strip()
    if not text:
        raise UnreadableAttachmentError(
            f"'{name}' has no extractable text (a scanned PDF needs OCR)."
        )

    truncated = len(text) > MAX_ATTACHMENT_CHARS
    if truncated:
        text = text[:MAX_ATTACHMENT_CHARS] + TRUNCATION_MARK
    log.info("attachment_extracted", filename=name, chars=len(text), truncated=truncated)
    return ExtractedAttachment(filename=name, text=text, truncated=truncated)


def _pdf_text(data: bytes) -> str:
    """Join the text of every PDF page, one blank line between pages."""
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)


def _docx_text(data: bytes) -> str:
    """Join the paragraphs and table cells of a Word document."""
    # NOTA: python-docx separa párrafos y tablas. Las tablas son frecuentes en
    # especificaciones (requisitos, presupuestos), así que también se leen:
    # cada fila sale como "celda | celda | celda".
    document = Document(io.BytesIO(data))
    lines = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))
    return "\n".join(lines)
