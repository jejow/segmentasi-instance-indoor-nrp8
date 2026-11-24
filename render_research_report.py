"""Render the study Markdown and its local figures as an A4 report."""
import argparse
from html import escape
from io import BytesIO
import re
from pathlib import Path

from PIL import Image as Raster, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, LongTable, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, TableStyle,
)


INK = colors.HexColor("#171717")
MUTED = colors.HexColor("#515151")
SHADE = colors.HexColor("#F1F1F1")
LINE = colors.HexColor("#CFCFCF")
MARGIN = 50
WIDTH = A4[0] - MARGIN * 2
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
LOCAL_FIGURE = re.compile(r"^<!-- figure: (.+?) \| (.+) -->$")
NUMBER = re.compile(r"[+-]?\d+(?:[.,]\d+)*(?:\s*%|\s+poin)?")


def fonts():
    font_dir = Path("C:/Windows/Fonts")
    family = ("arial.ttf", "arialbd.ttf", "ariali.ttf")
    if all((font_dir / name).is_file() for name in family):
        for suffix, name in zip(("", "Bold", "Italic"), family):
            pdfmetrics.registerFont(TTFont("Study" + suffix, str(font_dir / name)))
        pdfmetrics.registerFontFamily("Study", normal="Study", bold="StudyBold", italic="StudyItalic")
        return "Study", "StudyBold"
    return "Helvetica", "Helvetica-Bold"


def styles():
    regular, bold = fonts()
    body = ParagraphStyle("body", fontName=regular, fontSize=10.35, leading=14.45,
                          textColor=INK, spaceAfter=7, alignment=TA_LEFT,
                          allowWidows=0, allowOrphans=0)
    def variant(name, **properties):
        return ParagraphStyle(name, parent=body, **properties)
    return {
        "title": variant("title", fontName=bold, fontSize=34, leading=39,
                         spaceBefore=25, spaceAfter=15, keepWithNext=True),
        "meta": variant("meta", fontName=bold, fontSize=10, textColor=MUTED, spaceAfter=12, keepWithNext=True),
        "head": variant("head", fontName=bold, fontSize=18, leading=22,
                        spaceBefore=9, spaceAfter=9, keepWithNext=True),
        "subhead": variant("subhead", fontName=bold, fontSize=11.8, leading=15,
                           spaceBefore=8, spaceAfter=5, keepWithNext=True),
        "body": body,
        "list": variant("list", leftIndent=14, firstLineIndent=-14),
        "cell": variant("cell", fontSize=9, leading=12, spaceAfter=0),
        "cell_num": variant("cell_num", fontSize=9, leading=12, alignment=TA_RIGHT, spaceAfter=0),
        "header": variant("header", fontName=bold, fontSize=9, leading=12,
                          textColor=colors.white, spaceAfter=0),
        "header_num": variant("header_num", fontName=bold, fontSize=9, leading=12,
                              alignment=TA_RIGHT, textColor=colors.white, spaceAfter=0),
        "caption": variant("caption", fontSize=8.6, leading=11.4, textColor=MUTED, spaceAfter=9),
        "case": variant("case", fontName=bold, fontSize=11, leading=15, spaceAfter=4),
    }


def markup(line):
    line = escape(line.strip().replace("–", "-").replace("−", "-"))
    line = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", line)
    line = re.sub(r"`([^`]+)`", r"<font color='#343434'>\1</font>", line)

    def anchor(match):
        label, location = match.groups()
        if location.startswith(("https://", "http://")):
            return f'<link href="{location}" color="#212121"><u>{label}</u></link>'
        return f'<font color="#414141">{label}</font>'

    return LINK.sub(anchor, line)


def figure(path, limit=WIDTH, max_height=225):
    with Raster.open(path) as img:
        width, height = img.size
        raster = BytesIO()
        ImageOps.grayscale(img).save(raster, format="PNG")
        raster.seek(0)
    scale = min(limit / width, max_height / height)
    output = Image(raster, width=width * scale, height=height * scale)
    output.hAlign = "LEFT"
    return output


def data_table(rows, typography):
    columns = len(rows[0])
    lead_weight = 1.7 if columns >= 4 else 1.4
    unit = WIDTH / (columns - 1 + lead_weight)
    numerical = [all(NUMBER.fullmatch(row[col].strip()) for row in rows[1:]) for col in range(columns)]
    content = []
    for i, row in enumerate(rows):
        style = "header" if i == 0 else "cell"
        content.append([Paragraph(markup(cell), typography[style + ("_num" if numeric else "")])
                        for cell, numeric in zip(row, numerical, strict=True)])
    table = LongTable(content, colWidths=[lead_weight * unit, *(unit for _ in range(columns - 1))],
                      repeatRows=1, hAlign="LEFT", spaceAfter=9)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SHADE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), .5, LINE),
    ]))
    return table


def render(source, destination):
    typography, story = styles(), []
    lines = source.read_text(encoding="utf-8").splitlines()
    index, section = 0, ""
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        if line.startswith("# "):
            story.append(Paragraph(markup(line[2:]), typography["title"]))
        elif line.startswith(("**Visi Komputer", "**Studi komputasional")):
            story.append(Paragraph(markup(line), typography["meta"]))
            story.append(HRFlowable(width="100%", thickness=1.3, color=INK, spaceBefore=3, spaceAfter=14))
        elif line == "<!-- page-break -->":
            story.append(PageBreak())
        elif line.startswith("## "):
            section = line[3:]
            if section == "Bukti audit dan demo":
                story.append(PageBreak())
            story.append(Paragraph(markup(section), typography["head"]))
        elif line.startswith("### "):
            story.append(Paragraph(markup(line[4:]), typography["subhead"]))
        elif local := LOCAL_FIGURE.fullmatch(line):
            path, caption = local.groups()
            attachment = source.parent / path
            if attachment.is_file():
                height = {"Studi kasus kegagalan": 145, "Studi kasus dengan cahaya sulit": 116}.get(section, 155)
                story.append(KeepTogether([
                    figure(attachment, max_height=height),
                    Spacer(1, 5),
                    Paragraph(markup(caption), typography["caption"]),
                ]))
            else:
                story.append(Paragraph(markup(caption + " Bukti foto tersedia pada paket lokal."), typography["caption"]))
        elif line.startswith("!["):
            match = LINK.search(line)
            if match:
                caption, path = match.groups()
                story.append(KeepTogether([
                    figure(source.parent / path,
                           max_height=184 if section == "Bukti audit dan demo" else 192),
                    Spacer(1, 5),
                    Paragraph(markup(caption), typography["caption"]),
                ]))
        elif line.startswith("|"):
            rows = [line.strip("|").split("|")]
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_line = lines[index].strip()
                if not all(char in "|:- " for char in table_line):
                    rows.append(table_line.strip("|").split("|"))
                index += 1
            if section == "Lima contoh kegagalan":
                for label, evidence in rows[1:]:
                    match = LINK.search(label)
                    if match is None:
                        raise ValueError("Case row needs a local comparison figure.")
                    caption, path = match.groups()
                    story.append(KeepTogether([
                        Paragraph(markup(caption), typography["case"]),
                        figure(source.parent / path, max_height=142),
                        Spacer(1, 5), Paragraph(markup(evidence), typography["body"]),
                    ]))
            else:
                story.append(data_table(rows, typography))
        else:
            style = "list" if re.match(r"\d+\. ", line) else "caption" if section == "Referensi metode" else "body"
            story.append(Paragraph(markup(line), typography[style]))

    def footer(canvas, document):
        if document.page > 1:
            canvas.setFillColor(MUTED)
            canvas.setFont(typography["header"].fontName, 8)
            canvas.drawString(MARGIN, A4[1] - 32, "VISI KOMPUTER")
            canvas.drawRightString(A4[0] - MARGIN, A4[1] - 32, "CPMK 3  /  NRP 8")
            canvas.setStrokeColor(LINE)
            canvas.line(MARGIN, A4[1] - 40, A4[0] - MARGIN, A4[1] - 40)
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN, 41, A4[0] - MARGIN, 41)
        canvas.setFillColor(MUTED)
        canvas.setFont(typography["body"].fontName, 8)
        canvas.drawString(MARGIN, 26, "Segmentasi instance indoor")
        canvas.drawRightString(A4[0] - MARGIN, 26, f"{document.page:02d}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination), pagesize=A4, topMargin=52, bottomMargin=55,
        leftMargin=MARGIN, rightMargin=MARGIN,
        title="Segmentasi instance pada lingkungan indoor nyata",
        subject="Laporan eksperimen visi komputer NRP 8",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, default=Path("LAPORAN_RISET.md"), nargs="?")
    parser.add_argument("--output", type=Path, default=Path("output/pdf/laporan-segmentasi-indoor-nrp8.pdf"))
    options = parser.parse_args()
    render(options.source, options.output)
