"""The complete patient-history PDF.

Cover, contents (with page numbers), visit timeline, trend charts, per-
modality comparisons (latest vs previous, first vs latest), then every study
report in full, oldest first -- each built by the same per-modality spec as
its own PDF, so the history can never disagree with the individual reports.
"""
import io
from datetime import datetime
from html import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak,
                                PageTemplate, Paragraph, Spacer)
from reportlab.platypus.tableofcontents import TableOfContents

from records.catalog import MODALITY_LABELS
from records.compare import compare
from reporting.pdf import CONTENT_WIDTH, REPORT_STYLES, _data_table, report_story
from reporting.study_reports import REPORT_SPECS, form_meta
from webapp.metadata import PATIENT_FIELDS

ACCENT = "#0f766e"
H1 = ParagraphStyle("HistH1", parent=REPORT_STYLES["ReportTitle"], fontSize=16,
                    leading=20, spaceAfter=10, alignment=0)
TOC_L0 = ParagraphStyle("TOC0", fontName="Helvetica-Bold", fontSize=10.5,
                        leading=16, leftIndent=0)
TOC_L1 = ParagraphStyle("TOC1", fontName="Helvetica", fontSize=9.5,
                        leading=14, leftIndent=16)


class _HistoryDoc(BaseDocTemplate):
    """Registers chapter headings (H1) and study reports with the contents."""

    def __init__(self, buffer, patient, **kw):
        super().__init__(buffer, pagesize=A4, rightMargin=36, leftMargin=36,
                         topMargin=48, bottomMargin=42, **kw)
        self.patient = patient
        frame = Frame(self.leftMargin, self.bottomMargin, self.width,
                      self.height, id="body")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame],
                                            onPage=self._decorate)])

    def afterFlowable(self, flowable):
        level = getattr(flowable, "_toc_level", None)
        if level is not None:
            self.notify("TOCEntry", (level, flowable._toc_text, self.page))

    def _decorate(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setStrokeColor(colors.HexColor(ACCENT))
        canvas.setLineWidth(2)
        canvas.line(36, h - 30, w - 36, h - 30)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        p = self.patient
        canvas.drawString(36, h - 24, f"{p['name']} · {p['id']}")
        canvas.drawRightString(w - 36, h - 24, "Complete Patient History — confidential")
        canvas.drawString(36, 24, "Unified AI Diagnostic Platform — research prototype, not a medical device")
        canvas.drawRightString(w - 36, 24, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()


def _heading(text, level, style):
    para = Paragraph(escape(text), style)
    para._toc_level = level
    para._toc_text = text
    return para


def _png_flowable(png):
    from PIL import Image as PILImage
    w, h = PILImage.open(io.BytesIO(png)).size
    scale = min(CONTENT_WIDTH / w, 330 / h)
    return Image(io.BytesIO(png), width=w * scale, height=h * scale, hAlign="LEFT")


def _patient_rows(p):
    return [("Field", "Details"),
            ("Patient ID", p["id"]), ("Name", p["name"]),
            ("Date of birth", p.get("dob") or "—"),
            ("Age", "—" if p.get("age") is None else str(p["age"])),
            ("Sex", p.get("sex") or "—"), ("Phone", p.get("phone") or "—"),
            ("Address", p.get("address") or "—"),
            ("Standing notes", p.get("notes") or "—"),
            ("Registered", (p.get("created_at") or "")[:10])]


def _compare_rows(before, after):
    rows = [("Measure", f"{before['study_date']} → {after['study_date']}  ·  change / note")]
    for r in compare(before, after):
        if r["delta"] is not None:
            sign = "+" if r["delta"] > 0 else ""
            change = f"{sign}{r['delta']:.1f} {r['unit']}".strip()
        else:
            change = "changed" if r["changed"] else "unchanged"
        # Verdict/finding rows already show before -> after; their note would
        # only repeat it.
        note = f"  ·  {r['note']}" if r["note"] and r["key"] not in ("verdict", "headline") else ""
        rows.append((r["label"], f"{r['before']} → {r['after']}  ·  {change}{note}"))
    return rows


def build_history_pdf(patient, studies_full, trend_pngs, images_for):
    """BytesIO of the complete history.

    studies_full: full study dicts, any order. trend_pngs: {modality: png}.
    images_for(study) -> [(caption, path)] for that study's report.
    """
    ordered = sorted(studies_full, key=lambda s: (s["study_date"], s["created_at"]))
    buffer = io.BytesIO()
    doc = _HistoryDoc(buffer, patient, title=f"Patient history {patient['id']}",
                      author="Unified AI Diagnostic Platform")

    story = [
        Spacer(1, 40),
        Paragraph("COMPLETE PATIENT HISTORY", REPORT_STYLES["ReportTitle"]),
        Paragraph(escape(f"{patient['name']} · {patient['id']}"), REPORT_STYLES["ReportSubtitle"]),
        Paragraph(f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} · "
                  f"{len(ordered)} stud{'y' if len(ordered) == 1 else 'ies'}",
                  REPORT_STYLES["ReportSubtitle"]),
        Spacer(1, 22),
        _data_table(_patient_rows(patient), ACCENT),
        Spacer(1, 18),
    ]
    counts = {}
    for s in ordered:
        counts[s["modality"]] = counts.get(s["modality"], 0) + 1
    count_rows = [(MODALITY_LABELS[m], str(n)) for m, n in counts.items()]
    story.append(_data_table([("Modality", "Studies on record")]
                             + (count_rows or [("—", "No studies recorded")]), ACCENT))
    story.append(PageBreak())

    toc = TableOfContents()
    toc.levelStyles = [TOC_L0, TOC_L1]
    story += [Paragraph("Contents", H1), toc, PageBreak()]

    story.append(_heading("1. Visit Timeline", 0, H1))
    if ordered:
        story.append(_data_table(
            [("Date · Study", "Type · Verdict · Finding")] +
            [(f"{s['study_date']} · {s['id']}",
              f"{MODALITY_LABELS[s['modality']]} · {s.get('verdict') or 'NOT ASSESSED'}"
              f" · {s.get('headline') or '—'}") for s in ordered], ACCENT))
    else:
        story.append(Paragraph("No studies recorded.", REPORT_STYLES["ReportBody"]))

    if trend_pngs:
        story += [PageBreak(), _heading("2. Trends", 0, H1)]
        for modality, png in trend_pngs.items():
            story += [_heading(f"{MODALITY_LABELS[modality]} trends", 1,
                               REPORT_STYLES["SectionHeading"]),
                      _png_flowable(png), Spacer(1, 12)]

    by_mod = {}
    for s in ordered:
        by_mod.setdefault(s["modality"], []).append(s)
    comparable = {m: ss for m, ss in by_mod.items() if len(ss) >= 2}
    if comparable:
        story += [PageBreak(), _heading("3. Comparisons", 0, H1)]
        for modality, ss in comparable.items():
            label = MODALITY_LABELS[modality]
            story.append(_heading(f"{label}: Latest vs previous", 1,
                                  REPORT_STYLES["SectionHeading"]))
            story += [_data_table(_compare_rows(ss[-2], ss[-1]), ACCENT), Spacer(1, 12)]
            if len(ss) > 2:
                story.append(_heading(f"{label}: First vs latest", 1,
                                      REPORT_STYLES["SectionHeading"]))
                story += [_data_table(_compare_rows(ss[0], ss[-1]), ACCENT), Spacer(1, 12)]

    if ordered:
        story += [PageBreak(), _heading("4. Study Reports", 0, H1),
                  Paragraph("Each study's full report, oldest first, exactly as "
                            "issued for that visit.", REPORT_STYLES["ReportBody"])]
    for i, s in enumerate(ordered):
        spec = REPORT_SPECS[s["modality"]](s["report"])
        if i:
            story.append(PageBreak())
        else:
            story.append(Spacer(1, 12))
        story.append(_heading(f"{s['study_date']} · {MODALITY_LABELS[s['modality']]} · {s['id']}",
                              1, REPORT_STYLES["SectionHeading"]))
        images = [("image", caption, path) for caption, path in images_for(s)]
        story += report_story(spec["title"], spec["subtitle"], spec["accent"],
                              form_meta(s["report"]), PATIENT_FIELDS,
                              "Patient & Study Details",
                              spec["sections"] + images + spec["closing"],
                              spec["disclaimer"])

    doc.multiBuild(story)
    buffer.seek(0)
    return buffer

