"""Convert SUBMISSION.md to SUBMISSION.pdf using reportlab."""
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

MD_PATH = Path(__file__).parent.parent / "SUBMISSION.md"
PDF_PATH = Path(__file__).parent.parent / "SUBMISSION.pdf"


def make_styles():
    base = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=18, spaceAfter=8,
                             textColor=colors.HexColor("#1a1a2e")),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=14, spaceBefore=14,
                             spaceAfter=6, textColor=colors.HexColor("#16213e")),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11, spaceBefore=10,
                             spaceAfter=4, textColor=colors.HexColor("#0f3460")),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.5,
                               leading=14, spaceAfter=4),
        "code": ParagraphStyle("code", parent=base["Code"], fontSize=8,
                               fontName="Courier", backColor=colors.HexColor("#f4f4f4"),
                               leftIndent=12, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"], fontSize=9.5,
                                 leading=14, leftIndent=18, spaceAfter=2),
        "bold": ParagraphStyle("bold", parent=base["Normal"], fontSize=9.5,
                               leading=14, spaceAfter=4),
    }
    return styles


def parse_md(text, styles):
    elements = []
    lines = text.split("\n")
    i = 0
    in_code = False
    code_buf = []
    in_table = False
    table_buf = []

    while i < len(lines):
        line = lines[i]

        # Code block
        if line.strip().startswith("```"):
            if in_code:
                in_code = False
                code_text = "\n".join(code_buf)
                elements.append(Paragraph(code_text.replace("\n", "<br/>"), styles["code"]))
                code_buf = []
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            i += 1
            continue

        # Table
        if line.startswith("|"):
            table_buf.append(line)
            i += 1
            continue
        if table_buf:
            elements.append(render_table(table_buf, styles))
            elements.append(Spacer(1, 6))
            table_buf = []

        # Headings
        if line.startswith("### "):
            elements.append(Paragraph(escape(line[4:]), styles["h3"]))
        elif line.startswith("## "):
            elements.append(HRFlowable(width="100%", thickness=0.5,
                                       color=colors.HexColor("#cccccc"), spaceAfter=4))
            elements.append(Paragraph(escape(line[3:]), styles["h2"]))
        elif line.startswith("# "):
            elements.append(Paragraph(escape(line[2:]), styles["h1"]))
        elif line.startswith("---"):
            elements.append(Spacer(1, 4))
        elif line.startswith("- ") or line.startswith("* "):
            elements.append(Paragraph("• " + inline(line[2:]), styles["bullet"]))
        elif line.strip() == "":
            elements.append(Spacer(1, 4))
        else:
            elements.append(Paragraph(inline(line), styles["body"]))

        i += 1

    if table_buf:
        elements.append(render_table(table_buf, styles))

    return elements


def escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(s):
    s = escape(s)
    # Bold
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    # Italic
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    # Inline code
    s = re.sub(r"`(.+?)`", r'<font name="Courier" size="8">\1</font>', s)
    # Links
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="blue">\1</link>', s)
    return s


def render_table(lines, styles):
    rows = []
    for line in lines:
        if re.match(r"^\|[-| ]+\|$", line.strip()):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return Spacer(1, 1)

    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    from reportlab.platypus import Paragraph as P
    cell_style = ParagraphStyle("tc", fontSize=8.5, leading=12)
    header_style = ParagraphStyle("th", fontSize=8.5, leading=12, fontName="Helvetica-Bold")

    data = []
    for ri, row in enumerate(rows):
        st = header_style if ri == 0 else cell_style
        data.append([P(inline(c), st) for c in row])

    col_width = (6.5 * inch) / max_cols
    t = Table(data, colWidths=[col_width] * max_cols, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f9f9f9"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def main():
    text = MD_PATH.read_text()
    styles = make_styles()

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title="PawPrep — Project Submission",
        author="Nithin Reddy",
    )
    elements = parse_md(text, styles)
    doc.build(elements)
    print(f"PDF written → {PDF_PATH}")


if __name__ == "__main__":
    main()
