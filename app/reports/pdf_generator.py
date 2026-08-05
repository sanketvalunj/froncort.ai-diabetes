# Converts markdown reports into PDF format.
"""
MarkdownPDFGenerator — converts a pre-generated Markdown report to a PDF artifact.

Design goals
------------
- Additive only: called after the .md file is written; never changes Markdown content.
- Preserves all report information: headings, tables, bullet lists, blockquotes,
  horizontal rules, inline code, bold/italic, and plain paragraphs.
- Uses ReportLab's built-in Helvetica family (no external font files needed).
- Emoji and Unicode characters outside Latin-1 are replaced with clear ASCII
  equivalents so all content remains fully readable in the PDF.
- All table cells wrap automatically; no text escapes page margins.
- The 5-column criterion table uses weighted proportional column widths.
- Page numbers appear in the footer of every page.
- Table headers repeat on every page that a table spans.
- Alternate row shading and TOP-aligned cells throughout.

Output directory : <reports_dir>/../report_pdfs/
Filename         : <stem>.pdf  (mirrors the .md filename)
"""

import re
from pathlib import Path
from typing import List, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Page geometry ─────────────────────────────────────────────────────────────

_PAGE_W, _PAGE_H   = A4
_MARGIN            = 18 * mm
_BODY_W            = _PAGE_W - 2 * _MARGIN   # usable width ≈ 174 mm

# ── Unicode / emoji substitution ──────────────────────────────────────────────
# ReportLab's built-in Type-1 fonts (Helvetica) cover Latin-1 only.
# Replace emoji with short, self-explanatory ASCII tags so no information is lost.

_EMOJI: List[Tuple[str, str]] = [
    ("\u26a0\ufe0f", "[!]"),     # ⚠️
    ("\u2705",       "[OK]"),    # ✅
    ("\u274c",       "[NO]"),    # ❌
    ("\u2753",       "[?]"),     # ❓
    ("\U0001f50d",   "[R]"),     # 🔍
    ("\U0001f7e2",   "[OPEN]"),  # 🟢
    ("\U0001f534",   "[CLOSED]"),# 🔴
    ("\u26a1",       "[~~]"),    # ⚡
    ("\u26a0",       "[!]"),     # ⚠ (without variation selector)
    ("\u2014",       "--"),      # em dash
    ("\u2013",       "-"),       # en dash
    ("\u2018",       "'"),       # left single quote
    ("\u2019",       "'"),       # right single quote
    ("\u201c",       '"'),       # left double quote
    ("\u201d",       '"'),       # right double quote
    ("\u2026",       "..."),     # ellipsis
    ("\u00ae",       "(R)"),     # ®
    ("\u2122",       "(TM)"),    # ™
    ("\u2022",       "*"),       # bullet (fallback for any stray unicode bullets)
]


def _clean(text: str) -> str:
    """Replace non-Latin-1 characters with ASCII equivalents."""
    for char, replacement in _EMOJI:
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ── Inline Markdown → ReportLab XML ──────────────────────────────────────────

def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_inline(text: str) -> str:
    """
    Convert inline Markdown to ReportLab Paragraph XML.
    Order:
      1. Stash `code` spans so later regexes cannot touch their contents.
      2. XML-escape the remaining text.
      3. Apply **bold**, *italic*, _italic_.
      4. Restore code spans wrapped in Courier font tags.
    """
    stash: list = []

    def _stash_code(m: re.Match) -> str:
        inner       = _xml_escape(m.group(1))
        placeholder = f"\x00CODE{len(stash)}\x00"
        stash.append(f"<font name='Courier' fontSize='7'>{inner}</font>")
        return placeholder

    text = re.sub(r"`([^`]+)`", _stash_code, text)
    text = _xml_escape(text)
    text = re.sub(r"\*\*(.+?)\*\*",                              r"<b>\1</b>",  text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",        r"<i>\1</i>",  text)
    text = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)",          r"<i>\1</i>",  text)

    for idx, code_xml in enumerate(stash):
        text = text.replace(f"\x00CODE{idx}\x00", code_xml)
    return text


# ── Style registry ────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    base = getSampleStyleSheet()

    def ps(name: str, parent: str = "Normal", **kw) -> ParagraphStyle:
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        # ── Body text ────────────────────────────────────────────────────────
        "h1": ps("H1", "Heading1",
                 fontSize=15, spaceAfter=6, spaceBefore=14,
                 textColor=colors.HexColor("#1a1a2e"), leading=19),
        "h2": ps("H2", "Heading2",
                 fontSize=12, spaceAfter=4, spaceBefore=10,
                 textColor=colors.HexColor("#16213e"), leading=16),
        "h3": ps("H3", "Heading3",
                 fontSize=10.5, spaceAfter=3, spaceBefore=8,
                 textColor=colors.HexColor("#0f3460"), leading=14),
        "h4": ps("H4", "Heading4",
                 fontSize=9.5, spaceAfter=2, spaceBefore=6,
                 textColor=colors.HexColor("#0f3460"), leading=13),
        "body": ps("Body",
                   fontSize=9, leading=13, spaceAfter=3),
        "bullet": ps("Bullet",
                     fontSize=9, leading=13,
                     leftIndent=14, spaceAfter=2, bulletIndent=5),
        "code": ps("Code", "Code",
                   fontSize=7.5, leading=11, leftIndent=6,
                   fontName="Courier",
                   backColor=colors.HexColor("#f5f5f5")),
        "blockquote": ps("BQ",
                         fontSize=8.5, leading=12,
                         leftIndent=16, rightIndent=16, spaceAfter=5,
                         textColor=colors.HexColor("#444444"),
                         backColor=colors.HexColor("#f8f8f8"),
                         borderPadding=(4, 0, 4, 10)),
        # ── Table cells ──────────────────────────────────────────────────────
        "th": ps("TH",
                 fontSize=7.5, leading=10,
                 fontName="Helvetica-Bold",
                 textColor=colors.HexColor("#1a1a2e"),
                 alignment=TA_CENTER),
        "th_left": ps("THL",
                      fontSize=7.5, leading=10,
                      fontName="Helvetica-Bold",
                      textColor=colors.HexColor("#1a1a2e"),
                      alignment=TA_LEFT),
        "td": ps("TD",
                 fontSize=8, leading=11, wordWrap="LTR"),
        "td_center": ps("TDC",
                        fontSize=8, leading=11,
                        alignment=TA_CENTER, wordWrap="LTR"),
        "td_narrow": ps("TDN",
                        fontSize=7.5, leading=10, wordWrap="LTR"),
        # ── Page footer ──────────────────────────────────────────────────────
        "footer": ps("Footer",
                     fontSize=7, leading=9,
                     textColor=colors.HexColor("#888888"),
                     alignment=TA_CENTER),
    }


# ── Column-width profiles ─────────────────────────────────────────────────────
#
# The criterion table has 5 columns: Criterion | Status | Evaluator | Reasoning | Evidence
# Weights below are fractions of _BODY_W.  They are applied when the table
# header row matches the expected 5-column layout; all other tables get equal widths.

_CRITERION_COLS  = ["Criterion ID", "Criterion", "Status", "Evaluator",
                    "Reasoning", "Evidence Used", "Evidence"]
_COL_WEIGHTS_5   = [0.10, 0.12, 0.08, 0.35, 0.35]   # Crit | Status | Eval | Reason | Evidence
_COL_WEIGHTS_4   = [0.15, 0.15, 0.40, 0.30]
_COL_WEIGHTS_2   = [0.35, 0.65]


def _col_widths(header: List[str]) -> List[float]:
    """Return column widths (pts) proportional to content type."""
    n = len(header)
    if n == 5:
        return [w * _BODY_W for w in _COL_WEIGHTS_5]
    if n == 4:
        return [w * _BODY_W for w in _COL_WEIGHTS_4]
    if n == 2:
        return [w * _BODY_W for w in _COL_WEIGHTS_2]
    # Equal widths for any other column count
    return [_BODY_W / n] * n


# ── Table renderer ────────────────────────────────────────────────────────────

# Column headers whose cells should be centred
_CENTRE_COLS = {"Status", "Evaluator", "Score", "Count", "Field"}

# Max characters per cell — prevents a runaway cell from spanning multiple pages.
_MAX_CELL_CHARS = 350


def _cell_style(header: str, is_header: bool, styles: dict) -> ParagraphStyle:
    if is_header:
        return styles["th_left"] if header not in _CENTRE_COLS else styles["th"]
    return styles["td_center"] if header in _CENTRE_COLS else styles["td"]


def _render_table(header: List[str], rows: List[List[str]], styles: dict) -> Table:
    """Build a ReportLab Table with auto-wrapping cells and proportional widths."""
    col_widths = _col_widths(header)

    def make_cell(text: str, col_idx: int, is_header: bool = False) -> Paragraph:
        h_name = header[col_idx] if col_idx < len(header) else ""
        text   = _clean(text.strip())
        if not is_header and len(text) > _MAX_CELL_CHARS:
            text = text[:_MAX_CELL_CHARS - 3] + "..."
        style = _cell_style(h_name, is_header, styles)
        return Paragraph(_md_inline(text), style)

    data = [[make_cell(h, i, is_header=True) for i, h in enumerate(header)]]
    for row in rows:
        padded = row + [""] * max(0, len(header) - len(row))
        data.append([make_cell(padded[i], i) for i in range(len(header))])

    tbl = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,      # repeat header row on every page
        splitByRow=True,   # allow rows to break across pages
    )
    tbl.setStyle(TableStyle([
        # ── Header row ───────────────────────────────────────────────────
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#dde3f0")),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 7.5),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING",    (0, 0), (-1, 0), 5),
        # ── Body rows — alternating shading ──────────────────────────────
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f6fb")]),
        # ── Grid ─────────────────────────────────────────────────────────
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#c8ccd6")),
        ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#9aa0b0")),
        # ── Cell padding & alignment ──────────────────────────────────────
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
    ]))
    return tbl


# ── Markdown parser / flowable builder ───────────────────────────────────────

def _parse_table_row(line: str) -> List[str]:
    parts = line.split("|")
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return parts


def _is_separator_row(line: str) -> bool:
    cells = _parse_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-+:?", c.strip()) for c in cells)


def markdown_to_flowables(markdown: str, styles: dict) -> list:
    """
    Parse Markdown line-by-line into ReportLab flowables.

    Handled constructs:
      # H1 / ## H2 / ### H3 / #### H4
      ---  horizontal rule
      > blockquote
      - / * bullet item
      1.   numbered list item
      | Markdown tables |
      ``` fenced code blocks ```
      blank lines  → small spacer
      everything else → body paragraph
    """
    flowables: list = []
    lines   = markdown.splitlines()
    i, total = 0, len(lines)

    while i < total:
        line    = lines[i]
        stripped = line.strip()

        # ── Blank line ────────────────────────────────────────────────────
        if not stripped:
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # ── Heading ───────────────────────────────────────────────────────
        hm = re.match(r"^(#{1,4})\s+(.*)", stripped)
        if hm:
            level = len(hm.group(1))
            text  = _clean(hm.group(2))
            key   = f"h{level}" if level <= 4 else "h4"
            flowables.append(Paragraph(_md_inline(text), styles[key]))
            i += 1
            continue

        # ── Horizontal rule ───────────────────────────────────────────────
        if re.fullmatch(r"[-*_]{3,}", stripped):
            flowables.append(Spacer(1, 3))
            flowables.append(HRFlowable(
                width="100%", thickness=0.5,
                color=colors.HexColor("#b0b4be"),
                spaceAfter=6,
            ))
            i += 1
            continue

        # ── Blockquote ────────────────────────────────────────────────────
        if stripped.startswith(">"):
            # Collect consecutive blockquote lines
            bq_lines = []
            while i < total and lines[i].strip().startswith(">"):
                bq_lines.append(lines[i].strip().lstrip("> ").strip())
                i += 1
            text = " ".join(bq_lines)
            flowables.append(Paragraph(_md_inline(_clean(text)), styles["blockquote"]))
            continue

        # ── Fenced code block ─────────────────────────────────────────────
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < total and not lines[i].strip().startswith("```"):
                code_lines.append(_xml_escape(_clean(lines[i])))
                i += 1
            i += 1  # skip closing ```
            for cl in code_lines:
                flowables.append(Paragraph(cl, styles["code"]))
            flowables.append(Spacer(1, 4))
            continue

        # ── Bullet list item ──────────────────────────────────────────────
        if re.match(r"^[-*+]\s+", stripped):
            text = re.sub(r"^[-*+]\s+", "", stripped)
            flowables.append(Paragraph(
                "&#x2022; " + _md_inline(_clean(text)),
                styles["bullet"],
            ))
            i += 1
            continue

        # ── Numbered list item ────────────────────────────────────────────
        if re.match(r"^\d+\.\s+", stripped):
            text = re.sub(r"^\d+\.\s+", "", stripped)
            flowables.append(Paragraph(_md_inline(_clean(text)), styles["body"]))
            i += 1
            continue

        # ── Table ─────────────────────────────────────────────────────────
        if stripped.startswith("|"):
            table_lines = []
            while i < total and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= 2:
                header    = _parse_table_row(table_lines[0])
                body_rows = [
                    _parse_table_row(tl)
                    for tl in table_lines[1:]
                    if not _is_separator_row(tl)
                ]
                if header:
                    tbl = _render_table(header, body_rows, styles)
                    # KeepTogether wraps header + first data row so the header
                    # never appears alone at the bottom of a page.
                    flowables.append(KeepTogether([tbl, Spacer(1, 6)]))
            continue

        # ── Default: body paragraph ───────────────────────────────────────
        flowables.append(Paragraph(_md_inline(_clean(stripped)), styles["body"]))
        i += 1

    return flowables


# ── Page template with footer ─────────────────────────────────────────────────

def _make_footer_canvas(doc_ref):
    """
    Return an onPage callback that draws a page number footer.
    We keep a reference to the document so we can read doc.page at draw time.
    """
    def _draw_footer(canvas, doc):
        canvas.saveState()
        footer_text = f"Clinical Trial Pre-Screening Report  —  Page {doc.page}"
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawCentredString(
            _PAGE_W / 2,
            _MARGIN / 2,          # centred vertically in the bottom margin
            footer_text,
        )
        canvas.restoreState()

    return _draw_footer


# ── Public API ────────────────────────────────────────────────────────────────

def markdown_to_pdf(markdown: str, output_path: Path) -> Path:
    """
    Render *markdown* to a PDF file at *output_path*.
    Creates parent directories if they do not exist.
    Returns the resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN + 6 * mm,   # extra room for page-number footer
        title="Clinical Trial Pre-Screening Report",
        author="Clinical Trial Pre-Screening Assistant",
    )

    footer_cb = _make_footer_canvas(doc)
    flowables  = markdown_to_flowables(markdown, styles)

    if not flowables:
        flowables = [Paragraph("(empty report)", styles["body"])]

    doc.build(flowables, onFirstPage=footer_cb, onLaterPages=footer_cb)
    return output_path


def md_file_to_pdf(md_path: Path, pdf_dir: Path) -> Path:
    """
    Convert an existing Markdown report file to PDF.
    The PDF filename mirrors the Markdown stem: report.md → report.pdf.
    Written to *pdf_dir* (created if absent).
    """
    md_path  = Path(md_path)
    pdf_path = Path(pdf_dir) / (md_path.stem + ".pdf")
    markdown = md_path.read_text(encoding="utf-8")
    return markdown_to_pdf(markdown, pdf_path)
