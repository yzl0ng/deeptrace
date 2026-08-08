from __future__ import annotations

from io import BytesIO

import pytest

from app.ingestion.parsers import DocumentParseError, parse_document


def test_txt_requires_utf8_and_accepts_bom() -> None:
    blocks = parse_document(
        b"\xef\xbb\xbfSearchLab text",
        file_type="txt",
        filename="note.txt",
    )
    assert blocks[0].text == "SearchLab text"

    with pytest.raises(DocumentParseError, match="UTF-8"):
        parse_document(
            b"\xff\xfe\x00",
            file_type="txt",
            filename="bad.txt",
        )


def test_markdown_preserves_sections_and_code_blocks() -> None:
    blocks = parse_document(
        (
            "# SearchLab\n\nIntro.\n\n"
            "## Install\n\n```python\nprint('ok')\n```\n"
            "## Search\n\nUse RRF."
        ).encode(),
        file_type="md",
        filename="guide.md",
    )
    assert [block.section for block in blocks] == [
        "SearchLab",
        "Install",
        "Search",
    ]
    assert "```python" in blocks[1].text
    assert all(block.title == "SearchLab" for block in blocks)


def test_blank_document_is_rejected() -> None:
    with pytest.raises(DocumentParseError) as captured:
        parse_document(
            b" \n ",
            file_type="txt",
            filename="blank.txt",
        )
    assert captured.value.code == "no_extractable_text"


def test_text_pdf_preserves_page_number() -> None:
    blocks = parse_document(
        _text_pdf("SearchLab PDF evidence"),
        file_type="pdf",
        filename="evidence.pdf",
    )
    assert blocks[0].page_number == 1
    assert "SearchLab PDF evidence" in blocks[0].text


def _text_pdf(text: str) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    output = BytesIO()
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(
        f"BT /F1 12 Tf 40 240 Td ({escaped}) Tj ET".encode("ascii")
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(output)
    return output.getvalue()
