"""ReportLab document assembly for the three PDF reports.

Split out of app.py: this is a document renderer, not application logic. It
knows nothing about MRI, ECG or satellite imagery -- callers hand it a title,
an accent colour, a metadata block and a list of (kind, heading, payload)
sections.
"""
import io
from html import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from webapp.metadata import NOT_PROVIDED, metadata_rows
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)


REPORT_STYLES = getSampleStyleSheet()
REPORT_STYLES.add(ParagraphStyle(
    name="ReportTitle", parent=REPORT_STYLES["Title"],
    fontSize=18, leading=22, spaceAfter=2
))
REPORT_STYLES.add(ParagraphStyle(
    name="ReportSubtitle", parent=REPORT_STYLES["Normal"],
    fontSize=10, leading=14, textColor=colors.HexColor("#475569"), alignment=1
))
REPORT_STYLES.add(ParagraphStyle(
    name="SectionHeading", parent=REPORT_STYLES["Heading3"],
    fontSize=11.5, leading=14, spaceBefore=4, spaceAfter=6,
    textColor=colors.HexColor("#0f172a")
))
REPORT_STYLES.add(ParagraphStyle(
    name="ReportBody", parent=REPORT_STYLES["BodyText"],
    fontSize=10, leading=14.5
))
REPORT_STYLES.add(ParagraphStyle(
    name="Cell", parent=REPORT_STYLES["BodyText"],
    fontSize=9.5, leading=12.5, spaceBefore=0, spaceAfter=0
))
REPORT_STYLES.add(ParagraphStyle(
    name="Disclaimer", parent=REPORT_STYLES["Italic"],
    fontSize=8.5, leading=11.5, textColor=colors.HexColor("#64748b")
))

TABLE_WIDTHS = [200, 323]


def _cell(text, bold=False):
    """Escape a value and wrap it in a Paragraph so long text wraps in-cell."""
    safe = escape(str(text if text not in (None, "") else NOT_PROVIDED))
    if bold:
        safe = f"<b>{safe}</b>"
    return Paragraph(safe, REPORT_STYLES["Cell"])


def _data_table(rows, accent):
    """Build a two-column label/value table with a coloured header row."""
    data = [[_cell(rows[0][0], bold=True), _cell(rows[0][1], bold=True)]]
    data += [[_cell(label), _cell(value)] for label, value in rows[1:]]

    table = Table(data, colWidths=TABLE_WIDTHS, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(accent)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _header_footer(canvas, doc, accent, footer_text):
    """Draw the accent rule, page number and footer note on every page."""
    canvas.saveState()
    width, _ = A4

    canvas.setStrokeColor(colors.HexColor(accent))
    canvas.setLineWidth(2)
    canvas.line(36, doc.pagesize[1] - 30, width - 36, doc.pagesize[1] - 30)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#94a3b8"))
    canvas.drawString(36, 24, footer_text)
    canvas.drawRightString(width - 36, 24, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def build_pdf_report(title, subtitle, accent, meta, meta_fields,
                     meta_heading, sections, disclaimer, footer_text):
    """Assemble a consistently styled PDF and return it as a BytesIO buffer.

    sections: list of ("table", heading, [(label, value), ...])
              or        ("text",  heading, body_string)
    """
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=36, leftMargin=36, topMargin=48, bottomMargin=42,
        title=title, author="Unified AI Diagnostic Platform"
    )

    story = [
        Paragraph(escape(title), REPORT_STYLES["ReportTitle"]),
        Paragraph(escape(subtitle), REPORT_STYLES["ReportSubtitle"]),
        Spacer(1, 6),
        Paragraph(
            f"Report ID: <b>{escape(meta.get('report_id', NOT_PROVIDED))}</b> &nbsp;|&nbsp; "
            f"Generated: <b>{escape(meta.get('generated_at', NOT_PROVIDED))}</b>",
            REPORT_STYLES["ReportSubtitle"]
        ),
        Spacer(1, 16),
        Paragraph(escape(meta_heading), REPORT_STYLES["SectionHeading"]),
        _data_table([("Field", "Details")] + metadata_rows(meta, meta_fields), accent),
        Spacer(1, 16),
    ]

    for kind, heading, body in sections:
        story.append(Paragraph(escape(heading), REPORT_STYLES["SectionHeading"]))
        if kind == "table":
            story.append(_data_table(body, accent))
        else:
            story.append(Paragraph(escape(str(body)), REPORT_STYLES["ReportBody"]))
        story.append(Spacer(1, 14))

    story.append(Spacer(1, 6))
    story.append(Paragraph(escape(disclaimer), REPORT_STYLES["Disclaimer"]))

    def _decorate(canvas, doc):
        _header_footer(canvas, doc, accent, footer_text)

    document.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    buffer.seek(0)
    return buffer


