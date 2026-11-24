"""Render the study Markdown and its local figures as an A4 report."""
import argparse
from html import escape
import re
from pathlib import Path

from PIL import Image as Raster
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, LongTable, Paragraph, SimpleDocTemplate,
    Spacer, TableStyle,
)


INK = colors.HexColor("#182B3D")
MUTED = colors.HexColor("#516779")
BLUE = colors.HexColor("#205987")
TEAL = colors.HexColor("#086B63")
SHADE = colors.HexColor("#EFF5F8")
LINE = colors.HexColor("#D8E4EA")
MARGIN = 50
WIDTH = A4[0] - MARGIN * 2
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


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
    body = ParagraphStyle("body", fontName=regular, fontSize=10.2, leading=14.7,
                          textColor=INK, spaceAfter=7, alignment=TA_LEFT,
                          allowWidows=0, allowOrphans=0)
    def variant(name, **properties):
        return ParagraphStyle(name, parent=body, **properties)
    return {
        "title": variant("title", fontName=bold, fontSize=29, leading=34,
                         spaceAfter=12, keepWithNext=True),
        "meta": variant("meta", fontName=bold, textColor=TEAL, spaceAfter=16, keepWithNext=True),
        "head": variant("head", fontName=bold, textColor=BLUE, fontSize=16, leading=20,
                        spaceBefore=13, spaceAfter=6, keepWithNext=True),
        "body": body,
        "cell": variant("cell", fontSize=9, leading=12, spaceAfter=0),
        "header": variant("header", fontName=bold, fontSize=9, leading=12,
                          textColor=colors.white, spaceAfter=0),
        "caption": variant("caption", fontSize=8, leading=11, textColor=MUTED, spaceAfter=9),
        "case": variant("case", fontName=bold, fontSize=11, leading=15, textColor=TEAL, spaceAfter=4),
    }


def markup(line):
    line = escape(line.strip().replace("–", "-").replace("−", "-"))
    line = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", line)
    line = re.sub(r"`([^`]+)`", r"<font color='#086b63'>\1</font>", line)

    def anchor(match):
        label, location = match.groups()
        if location.startswith(("https://", "http://")):
            return f'<link href="{location}" color="#205987">{label}</link>'
        return f'<font color="#205987">{label}</font>'

    return LINK.sub(anchor, line)


def figure(path, limit=WIDTH, max_height=225):
    with Raster.open(path) as img:
        width, height = img.size
    scale = min(limit / width, max_height / height)
    output = Image(str(path), width=width * scale, height=height * scale)
    output.hAlign = "LEFT"
    return output


def data_table(rows, typography):
    columns = len(rows[0])
    lead_weight = 1.7 if columns >= 4 else 1.4
    unit = WIDTH / (columns - 1 + lead_weight)
    content = [
        [Paragraph(markup(cell), typography["header" if i == 0 else "cell"]) for cell in row]
        for i, row in enumerate(rows)
    ]
    table = LongTable(content, colWidths=[lead_weight * unit, *(unit for _ in range(columns - 1))],
                      repeatRows=1, hAlign="LEFT", spaceAfter=9)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SHADE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
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
        elif line.startswith("**Studi komputasional"):
            story.append(Paragraph(markup(line), typography["meta"]))
        elif line.startswith("## "):
            section = line[3:]
            story.append(Paragraph(markup(section), typography["head"]))
        elif line.startswith("!["):
            match = LINK.search(line)
            if match:
                caption, path = match.groups()
                story.append(KeepTogether([
                    figure(source.parent / path,
                           max_height=210 if section == "Bukti audit dan demo" else 225),
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
                        figure(source.parent / path, max_height=156),
                        Spacer(1, 5), Paragraph(markup(evidence), typography["body"]),
                    ]))
            else:
                story.append(data_table(rows, typography))
        else:
            story.append(Paragraph(markup(line), typography["body"]))

    def footer(canvas, document):
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN, 41, A4[0] - MARGIN, 41)
        canvas.setFillColor(MUTED)
        canvas.setFont(typography["body"].fontName, 8)
        canvas.drawString(MARGIN, 26, "Segmentasi instance indoor  |  NRP 8")
        canvas.drawRightString(A4[0] - MARGIN, 26, str(document.page))

    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination), pagesize=A4, topMargin=48, bottomMargin=57,
        leftMargin=MARGIN, rightMargin=MARGIN,
        title="Segmentasi instance indoor dan transfer ke foto ponsel",
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
