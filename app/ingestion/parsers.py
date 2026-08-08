from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

from app.ingestion.models import ParsedBlock


class DocumentParseError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def parse_document(
    content: bytes,
    *,
    file_type: str,
    filename: str,
    title_override: str | None = None,
) -> list[ParsedBlock]:
    title = (title_override or Path(filename).stem).strip()
    if file_type == "txt":
        text = _decode_utf8(content)
        return _text_blocks(text, title)
    if file_type == "md":
        text = _decode_utf8(content)
        return _markdown_blocks(text, title)
    if file_type == "pdf":
        return _pdf_blocks(content, title)
    raise DocumentParseError(
        "unsupported_file_type",
        "No parser is available for this file type.",
    )


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(
        re.sub(r"[ \t]+", " ", line).strip()
        for line in normalized.splitlines()
    )
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def normalized_document_content(blocks: list[ParsedBlock]) -> str:
    return "\n\n".join(
        (
            f"[page={block.page_number or ''};section={block.section or ''}]\n"
            f"{normalize_text(block.text)}"
        )
        for block in blocks
        if normalize_text(block.text)
    )


def _decode_utf8(content: bytes) -> str:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DocumentParseError(
            "invalid_utf8",
            "TXT and Markdown files must use UTF-8 encoding.",
        ) from error
    normalized = normalize_text(text)
    if not normalized:
        raise DocumentParseError(
            "no_extractable_text",
            "The file does not contain extractable text.",
        )
    return normalized


def _text_blocks(text: str, title: str) -> list[ParsedBlock]:
    return [ParsedBlock(title=title, text=text)]


def _markdown_blocks(text: str, fallback_title: str) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    current_heading: str | None = None
    document_title = fallback_title
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal buffer
        content = normalize_text("\n".join(buffer))
        if content:
            blocks.append(
                ParsedBlock(
                    title=document_title,
                    section=current_heading,
                    text=content,
                )
            )
        buffer = []

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            buffer.append(line)
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading and not in_fence:
            flush()
            current_heading = normalize_text(heading.group(2))
            if len(heading.group(1)) == 1 and document_title == fallback_title:
                document_title = current_heading
            continue
        buffer.append(line)
    flush()
    if not blocks:
        raise DocumentParseError(
            "no_extractable_text",
            "The Markdown file does not contain extractable text.",
        )
    return blocks


def _pdf_blocks(content: bytes, title: str) -> list[ParsedBlock]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as error:
        raise DocumentParseError(
            "pdf_parser_unavailable",
            "PDF support is not installed.",
        ) from error
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            try:
                if reader.decrypt("") == 0:
                    raise DocumentParseError(
                        "encrypted_pdf",
                        "Encrypted PDFs are not supported.",
                    )
            except DocumentParseError:
                raise
            except Exception as error:
                raise DocumentParseError(
                    "encrypted_pdf",
                    "Encrypted PDFs are not supported.",
                ) from error
        blocks = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                blocks.append(
                    ParsedBlock(
                        title=title,
                        page_number=page_number,
                        text=text,
                    )
                )
    except DocumentParseError:
        raise
    except (PdfReadError, ValueError, OSError) as error:
        raise DocumentParseError(
            "invalid_pdf",
            "The PDF is damaged or cannot be parsed.",
        ) from error
    except Exception as error:
        raise DocumentParseError(
            "invalid_pdf",
            "The PDF could not be parsed.",
        ) from error
    if not blocks:
        raise DocumentParseError(
            "scanned_pdf_or_no_text",
            "The PDF has no extractable text; OCR is not supported yet.",
        )
    return blocks
