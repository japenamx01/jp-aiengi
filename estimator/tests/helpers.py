"""Build small PDF and Word documents in memory for attachment tests.

NOTA: generar los documentos en el propio test evita guardar binarios en el
repo y deja a la vista qué texto contienen: el test sabe exactamente qué debe
aparecer en el prompt después de la extracción.
"""

from __future__ import annotations

import io

from docx import Document


def make_pdf(*lines: str) -> bytes:
    """Return a one-page PDF whose text layer contains ``lines`` (ASCII only)."""

    # NOTA: un PDF mínimo son 5 objetos (catálogo, árbol de páginas, página,
    # contenido y fuente) más una tabla "xref" con la posición en bytes de cada
    # uno. El contenido usa operadores de texto: BT/ET abren y cierran el
    # bloque, Tj pinta una cadena y T* salta a la línea siguiente.
    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    text_ops = " T* ".join(f"({escape(line)}) Tj" for line in lines)
    content = f"BT /F1 12 Tf 14 TL 72 720 Td {text_ops} ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n%s\nendobj\n" % (number, body))
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    )
    return out.getvalue()


def make_docx(*paragraphs: str, table: list[list[str]] | None = None) -> bytes:
    """Return a .docx with ``paragraphs`` and an optional table."""
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    if table:
        grid = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                grid.cell(r, c).text = value
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()
