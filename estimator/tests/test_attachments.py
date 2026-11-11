"""Unit tests of the local text extraction (path B)."""

import pytest

from app.services import attachments
from app.services.attachments import (
    UnreadableAttachmentError,
    UnsupportedAttachmentError,
    extract_attachment,
)
from tests.helpers import make_docx, make_pdf


def test_pdf_text_is_extracted() -> None:
    result = extract_attachment("spec.pdf", make_pdf("Offline mode required", "Budget: 20k"))

    assert result.filename == "spec.pdf"
    assert "Offline mode required" in result.text
    assert "Budget: 20k" in result.text
    assert not result.truncated


def test_docx_paragraphs_and_tables_are_extracted() -> None:
    data = make_docx("Scope", table=[["Module", "Hours"], ["Billing", "40"]])

    text = extract_attachment("proposal.docx", data).text

    assert "Scope" in text
    assert "Billing | 40" in text


def test_extension_check_is_case_insensitive_and_strips_paths() -> None:
    result = extract_attachment("../../uploads/SPEC.PDF", make_pdf("Hello"))
    assert result.filename == "SPEC.PDF"


@pytest.mark.parametrize("filename", ["notes.txt", "diagram.png", "old.doc", "no_extension"])
def test_unsupported_types_are_rejected(filename: str) -> None:
    with pytest.raises(UnsupportedAttachmentError):
        extract_attachment(filename, b"data")


@pytest.mark.parametrize("filename", ["broken.pdf", "broken.docx"])
def test_corrupt_files_are_unreadable(filename: str) -> None:
    with pytest.raises(UnreadableAttachmentError):
        extract_attachment(filename, b"this is not a real document")


def test_pdf_without_text_is_unreadable() -> None:
    with pytest.raises(UnreadableAttachmentError, match="no extractable text"):
        extract_attachment("scan.pdf", make_pdf())


def test_long_text_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attachments, "MAX_ATTACHMENT_CHARS", 10)

    result = extract_attachment("long.pdf", make_pdf("A" * 50))

    assert result.truncated
    assert result.text == "A" * 10 + attachments.TRUNCATION_MARK


def test_too_large_files_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attachments, "MAX_ATTACHMENT_BYTES", 100)
    with pytest.raises(UnreadableAttachmentError, match="limit"):
        extract_attachment("big.pdf", make_pdf("x" * 200))
