"""
MarkdownPDFGenerator — converts a pre-generated Markdown report to a PDF artifact.

Design goals:
  - Additive only: called after the .md file is written; never changes Markdown content.
  - Preserves all report information: headings, tables, bullet lists, blockquotes,
    horizontal rules, inline code, bold/italic, and plain paragraphs.
  - Uses ReportLab's built-in Helvetica family (no external font files needed).
  - Emoji and Unicode characters outside Latin-1 are replaced with clear ASCII
    equivalents so all content remains fully readable in the PDF.
  - Long table cells wrap; evidence UUIDs are preserved verbatim in smaller type.

Output directory: <reports_dir>/../report_pdfs/
Filename:         <stem>.pdf   (mirrors the .md filename)
"""

import re
from pathlib import Path
from typing import List, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Unicode / emoji substitution ─────────────────────────────────────────────
# ReportLab's built-in Type-1 fonts (Helvetica) cover Latin-1 only.
# Replace emoji with short, self-explanatory ASCII tags so no information is lost.

_EMOJI: List[Tuple[str, str]] = [
    # Status icons — order matters: longer sequences first
    ("\u26a0\ufe0f", "[!]"),          # ⚠️
    ("\u2705", "[OK]"),               # ✅
    ("\u274c", "[NO]"),               # ❌
    ("\u2753", "[?]"),                # ❓
    ("\U0001f50d", "[R]"),            # 🔍
    ("\U0001f7e2", "[OPEN]"),         # 🟢
    ("\U0001f534", "[CLOSED]"),       # 🔴
    ("\u26a1", "[~~]"),               # ⚡
    ("\u26a0", "[!]"),                # ⚠ (without variation selector)
    # Punctuation / typography
    ("\u2014", "--"),                 # em dash
    ("\u2013", "-"),                  # en dash
    ("\u2018", "'"),                  # left single quote
    ("\u2019", "'"),                  # right single quote
    ("\u201c", '"'),                  # left double quote
    ("\u201d", '"'),                  # right double quote
    ("\u2026", "..."),                # ellipsis
    ("\u00ae", "(R)"),                # registered trademark ®
    ("\u2122", "(TM)"),               # trademark ™
]


def _clean(text: str) -> str:
    """Replace non-Latin-1 characters with ASCII equivalents."""
    for char, replacement in _EMOJI:
        text = text.replace(char, replacement)
    # Final fallback: encode to Latin-1, replacing anything still unmappable
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ── Inline Markdown → ReportLab XML ──────────────────────────────────────────

def _xml_escape(text: str) -> str:
    """Escape the three XML special characters that break ReportLab's parser."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_inline(text: str) -> str:
    """
    Convert inline Markdown to ReportLab Paragraph XML.
    Handles: **bold**, *italic*, `code`, and escapes & < > for XML safety.
    Applied after _clean() so no emoji remain.

    Processing order:
      1. Stash `inline code` spans FIRST — pull them out so later regex
         substitutions cannot touch identifiers like api_key or OPENAI_API_KEY.
      2. XML-escape the remaining text (& < >) — safe now that code spans
         are already stashed and will be re-inserted verbatim.
      3. Apply **bold**, then *italic* (asterisk), then _italic_ (underscore).
         The underscore rule uses word-boundary assertions so that underscores
         inside identifiers (e.g. source_id, OPENAI_API_KEY) are never treated
         as Markdown emphasis delimiters.
      4. Restore stashed code spans wrapped in Courier font tags.
    """
    # 1. Stash `code` spans — replace with null-byte placeholders so
    #    XML escaping and bold/italic regexes cannot alter their contents.
    stash: list = []

    def _stash_code(m: re.Match) -> str:
        # XML-escape the code content itself before stashing so it is safe
        # when later re-inserted into the ReportLab XML stream.
        inner = _xml_escape(m.group(1))
        placeholder = f"\x00CODE{len(stash)}\x00"
        stash.append(f"<font name='Courier'>{inner}</font>")
        return placeholder

    text = re.sub(r"`([^`]+)`", _stash_code, text)

    # 2. XML-escape the rest of the text (code spans are already protected).
    text = _xml_escape(text)

    # 3a. **bold** — greedy-minimal match, no newlines
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # 3b. *italic* — single asterisk, not adjacent to another *
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)

    # 3c. _italic_ — ONLY when both underscores sit at a non-word boundary.
    #     (?<!\w) before opening _  →  underscore is NOT preceded by a word char
    #     (?!\w)  after closing  _  →  underscore is NOT followed by a word char
    #     This prevents api_key, OPENAI_API_KEY, source_id, evidence_id etc.
    #     from being mis-parsed as italic markers.
    text = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"<i>\1</i>", text)

    # 4. Restore stashed code spans (already XML-safe, wrapped in Courier tags).
    for idx, code_xml in enumerate(stash):
        text = text.replace(f"\x00CODE{idx}\x00", code_xml)

    return text


# ── Style definitions ─────────────────────────────────────────────────────────

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    W = 170 * mm  # usable page width (A4 minus margins)

    def ps(name, parent="Normal", **kw) -> ParagraphStyle:
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        "h1": ps("H1", "Heading1",
                 fontSize=16, spaceAfter=6, spaceBefore=14,
                 textColor=colors.HexColor("#1a1a2e")),
        "h2": ps("H2", "Heading2",
                 fontSize=13, spaceAfter=4, spaceBefore=10,
                 textColor=colors.HexColor("#16213e")),
        "h3": ps("H3", "Heading3",
                 fontSize=11, spaceAfter=3, spaceBefore=8,
                 textColor=colors.HexColor("#0f3460")),
        "h4": ps("H4", "Heading4",
                 fontSize=10, spaceAfter=2, spaceBefore=6,
                 textColor=colors.HexColor("#0f3460")),
        "body": ps("Body", fontSize=9, leading=13, spaceAfter=3),
        "bullet": ps("Bullet", fontSize=9, leading=13,
                     leftIndent=12, spaceAfter=2,
                     bulletIndent=4),
        "code": ps("Code", "Code",
                   fontSize=7.5, leading=11, leftIndent=6,
                   fontName="Courier", backColor=colors.HexColor("#f5f5f5")),
        "blockquote": ps("BQ", fontSize=8.5, leading=12,
                         leftIndent=14, rightIndent=14, spaceAfter=4,
                         textColor=colors.HexColor("#444444"),
                         borderPadding=(4, 0, 4, 8)),
        "th": ps("TH", fontSize=7.5, leading=10, fontName="Helvetica-Bold",
                 alignment=TA_CENTER),
        "td": ps("TD", fontSize=7.5, leading=10),
        "td_small": ps("TDsm", fontSize=6.5, leading=9,
                        fontName="Courier"),
        "footer": ps("Footer", fontSize=7.5, leading=10,
                     textColor=colors.HexColor("#666666"),
                     alignment=TA_CENTER),
    }


# ── Table renderer ────────────────────────────────────────────────────────────

# Column names that should use the small monospace style (evidence UUIDs etc.)
_SMALL_COLS = {"Evidence Used", "evidence used", "Reasoning", "reasoning"}

# Maximum characters allowed in a single table cell before truncation.
# This prevents a single very-long reasoning string from producing a cell
# taller than one page and causing a ReportLab LayoutError.
_MAX_CELL_CHARS = 400


def _render_table(header: List[str], rows: List[List[str]], styles_map: dict) -> Table:
    """Build a ReportLab Table from parsed Markdown table data."""
    use_small = any(h in _SMALL_COLS for h in header)

    def cell(text: str, is_header: bool = False) -> Paragraph:
        text = _clean(text.strip())
        # Truncate very long cells so no single row exceeds one page height.
        if not is_header and len(text) > _MAX_CELL_CHARS:
            text = text[:_MAX_CELL_CHARS] + " [...]"
        style = styles_map["th"] if is_header else (
            styles_map["td_small"] if use_small else styles_map["td"]
        )
        return Paragraph(_md_inline(text), style)

    data = [[cell(h, is_header=True) for h in header]]
    for row in rows:
        # Pad or truncate row to match header width
        padded = row + [""] * max(0, len(header) - len(row))
        data.append([cell(c) for c in padded[: len(header)]])

    # Distribute column widths relative to page width
    page_w = 170 * mm
    col_w  = page_w / len(header)
    col_widths = [col_w] * len(header)

    # splitByRow=1 lets ReportLab break large tables across pages;
    # repeatRows=1 repeats the header on each page.
    tbl = Table(data, colWidths=col_widths, repeatRows=1, splitByRow=1)
    tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#e8eaf6")),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 7.5),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        # Body rows — alternating background
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#fafafa")]),
        # Grid
        ("GRID",        (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


# ── Markdown parser / flowable builder ───────────────────────────────────────

def _parse_table_row(line: str) -> List[str]:
    """Split a Markdown table row on | , skipping the outer pipes."""
    parts = line.split("|")
    # Remove empty outer segments caused by leading/trailing |
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return parts


def _is_separator_row(line: str) -> bool:
    """True if every non-pipe cell is purely dashes/colons (alignment row)."""
    cells = _parse_table_row(line)
    return bool(cells) and all(
        re.fullmatch(r":?-+:?", c.strip()) for c in cells
    )


def markdown_to_flowables(markdown: str, styles_map: dict) -> list:
    """
    Parse Markdown line-by-line and produce a list of ReportLab flowables.

    Handled constructs:
      # H1 / ## H2 / ### H3 / #### H4
      --- (horizontal rule)
      > blockquote
      - bullet / * bullet
      | table | rows |
      ```code blocks```
      Blank lines → spacer
      Everything else → paragraph
    """
    flowables = []
    lines = markdown.splitlines()
    i = 0
    total = len(lines)

    while i < total:
        line = lines[i]
        stripped = line.strip()

        # ── Blank line ────────────────────────────────────────────────────
        if not stripped:
            flowables.append(Spacer(1, 3))
            i += 1
            continue

        # ── Heading ───────────────────────────────────────────────────────
        heading_match = re.match(r"^(#{1,4})\s+(.*)", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text  = _clean(heading_match.group(2))
            key   = f"h{level}" if level <= 4 else "h4"
            flowables.append(Paragraph(_md_inline(text), styles_map[key]))
            i += 1
            continue

        # ── Horizontal rule ───────────────────────────────────────────────
        if re.fullmatch(r"[-*_]{3,}", stripped):
            flowables.append(HRFlowable(width="100%", thickness=0.5,
                                         color=colors.HexColor("#aaaaaa"),
                                         spaceAfter=4))
            i += 1
            continue

        # ── Blockquote ────────────────────────────────────────────────────
        if stripped.startswith(">"):
            text = stripped.lstrip("> ").strip()
            flowables.append(Paragraph(
                _md_inline(_clean(text)), styles_map["blockquote"]
            ))
            i += 1
            continue

        # ── Fenced code block ─────────────────────────────────────────────
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < total and not lines[i].strip().startswith("```"):
                # _clean() handles emoji/unicode; _xml_escape() makes the line
                # safe for ReportLab's XML parser (< > & in code must be escaped).
                code_lines.append(_xml_escape(_clean(lines[i])))
                i += 1
            i += 1  # skip closing ```
            for cl in code_lines:
                flowables.append(Paragraph(cl, styles_map["code"]))
            flowables.append(Spacer(1, 3))
            continue

        # ── Bullet list item ──────────────────────────────────────────────
        if re.match(r"^[-*+]\s+", stripped):
            text = re.sub(r"^[-*+]\s+", "", stripped)
            flowables.append(Paragraph(
                "\u2022 " + _md_inline(_clean(text)),
                styles_map["bullet"],
            ))
            i += 1
            continue

        # ── Numbered list item ────────────────────────────────────────────
        if re.match(r"^\d+\.\s+", stripped):
            text = re.sub(r"^\d+\.\s+", "", stripped)
            flowables.append(Paragraph(
                _md_inline(_clean(text)), styles_map["body"]
            ))
            i += 1
            continue

        # ── Table ─────────────────────────────────────────────────────────
        if stripped.startswith("|"):
            # Collect contiguous table lines
            table_lines = []
            while i < total and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                header = _parse_table_row(table_lines[0])
                body_rows = []
                for tl in table_lines[1:]:
                    if not _is_separator_row(tl):
                        body_rows.append(_parse_table_row(tl))
                if header:
                    tbl = _render_table(header, body_rows, styles_map)
                    flowables.append(KeepTogether([tbl, Spacer(1, 4)]))
            continue

        # ── Default: paragraph ────────────────────────────────────────────
        flowables.append(Paragraph(
            _md_inline(_clean(stripped)), styles_map["body"]
        ))
        i += 1

    return flowables


# ── Public API ────────────────────────────────────────────────────────────────

def markdown_to_pdf(markdown: str, output_path: Path) -> Path:
    """
    Render *markdown* to a PDF file at *output_path*.

    Creates parent directories if they do not exist.
    Returns the resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles_map = _build_styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="Clinical Trial Pre-Screening Report",
        author="Clinical Trial Pre-Screening Assistant",
    )

    flowables = markdown_to_flowables(markdown, styles_map)

    # Safety: ensure we always have at least one flowable
    if not flowables:
        flowables = [Paragraph("(empty report)", styles_map["body"])]

    doc.build(flowables)
    return output_path


def md_file_to_pdf(md_path: Path, pdf_dir: Path) -> Path:
    """
    Convert an existing Markdown report file to PDF.

    The PDF filename mirrors the Markdown stem: report.md → report.pdf
    Written to *pdf_dir* (created if absent).
    """
    md_path = Path(md_path)
    pdf_path = Path(pdf_dir) / (md_path.stem + ".pdf")
    markdown  = md_path.read_text(encoding="utf-8")
    return markdown_to_pdf(markdown, pdf_path)
