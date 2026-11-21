"""python-pptx helpers that reproduce the course slide template (16:9, white, navy text,
navy footer bar 'Big Data Analytics Techniques and Applications | Fall 2026  Page N  Assignment 2')."""
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

NAVY = RGBColor(0x00, 0x00, 0x89)       # template text colour
BAR = RGBColor(0x24, 0x3E, 0x75)        # template footer bar
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xE8, 0xEC, 0xF6)
GOOD = RGBColor(0xE3, 0xF4, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x22, 0x22, 0x22)
FONT = "Arial"
W, H = Emu(9144000), Emu(5143500)       # 10 in x 5.625 in, as the template


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def save(self, path):
        self.prs.save(path)

    # ---------------------------------------------------------------- basics
    def slide(self, notes=None):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        self._footer(s)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def _footer(self, s):
        h = Inches(0.19)
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, H - h, W, h)
        bar.fill.solid(); bar.fill.fore_color.rgb = BAR; bar.line.fill.background()
        for text, left, width, align in (
                ("Big Data Analytics Techniques and Applications  |  Fall 2026", Inches(0.05), Inches(4.0), PP_ALIGN.LEFT),
                (f"Page {self.n}", Inches(4.0), Inches(1.0), PP_ALIGN.LEFT),
                ("Assignment 2", Inches(8.0), Inches(1.95), PP_ALIGN.RIGHT)):
            tb = s.shapes.add_textbox(left, H - h, width, h)
            tf = tb.text_frame; tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]; p.alignment = align
            r = p.add_run(); r.text = text; r.font.size = Pt(8); r.font.color.rgb = WHITE; r.font.name = FONT

    def text(self, s, left, top, width, height, paras, size=14, color=NAVY, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, fill=None, spacing=1.0):
        """paras: list of str or (str, dict(level=, bold=, size=, color=, italic=)) items. A str
        may contain **bold** segments."""
        tb = s.shapes.add_textbox(left, top, width, height)
        if fill is not None:
            tb.fill.solid(); tb.fill.fore_color.rgb = fill
        tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.06); tf.margin_top = tf.margin_bottom = Inches(0.03)
        first = True
        for item in paras:
            txt, opt = (item, {}) if isinstance(item, str) else item
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.alignment = opt.get("align", align)
            p.line_spacing = spacing
            lvl = opt.get("level", 0)
            bullet = opt.get("bullet", lvl > 0 or opt.get("b", False))
            prefix = ("•  " if lvl <= 1 else "–  ") if bullet else ""
            if lvl > 1:
                prefix = "    " * (lvl - 1) + prefix
            parts = (prefix + txt).split("**")
            for i, part in enumerate(parts):
                if not part:
                    continue
                r = p.add_run(); r.text = part
                f = r.font; f.name = FONT; f.size = Pt(opt.get("size", size - (1 if lvl > 1 else 0)))
                f.bold = opt.get("bold", bold) or (i % 2 == 1)
                f.italic = opt.get("italic", False)
                f.color.rgb = opt.get("color", color)
            if opt.get("space_after") is not None:
                p.space_after = Pt(opt["space_after"])
        return tb

    def header(self, s, title, question=None, underline=True):
        """Template header: '(3%) Question 2' bold underlined + the question in navy."""
        tb = s.shapes.add_textbox(Inches(0.45), Inches(0.18), Inches(9.1), Inches(0.5))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run(); r.text = title
        r.font.name = FONT; r.font.size = Pt(22); r.font.bold = True; r.font.underline = underline
        r.font.color.rgb = NAVY
        if question:
            p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = question
            r2.font.name = FONT; r2.font.size = Pt(15); r2.font.color.rgb = NAVY
        return tb

    def image(self, s, path, left, top, width=None, height=None):
        """Place an image inside the box (left, top, width, height) keeping its aspect ratio."""
        iw, ih = Image.open(path).size
        if width is not None and height is not None:
            scale = min(width / iw, height / ih)
            w, h = int(iw * scale), int(ih * scale)
            left, top = left + (width - w) // 2, top + (height - h) // 2
            return s.shapes.add_picture(str(path), left, top, w, h)
        return s.shapes.add_picture(str(path), left, top, width, height)

    def table(self, s, rows, left, top, width, height, col_widths=None, size=10, header_fill=BAR,
              highlight=(), bold_rows=(), align_right_from=1):
        nr, nc = len(rows), len(rows[0])
        gt = s.shapes.add_table(nr, nc, left, top, width, height).table
        if col_widths:
            tot = sum(col_widths)
            for j, cw in enumerate(col_widths):
                gt.columns[j].width = int(width * cw / tot)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                c = gt.cell(i, j)
                c.margin_left = c.margin_right = Inches(0.04); c.margin_top = c.margin_bottom = Inches(0.01)
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
                tf = c.text_frame; tf.word_wrap = True
                p = tf.paragraphs[0]
                p.alignment = PP_ALIGN.RIGHT if (j >= align_right_from and i > 0) else PP_ALIGN.LEFT
                r = p.add_run(); r.text = str(val)
                r.font.name = FONT; r.font.size = Pt(size)
                if i == 0:
                    r.font.bold = True; r.font.color.rgb = WHITE
                    c.fill.solid(); c.fill.fore_color.rgb = header_fill
                else:
                    r.font.color.rgb = BLACK
                    r.font.bold = i in bold_rows
                    c.fill.solid()
                    c.fill.fore_color.rgb = GOOD if i in highlight else (WHITE if i % 2 else LIGHT)
        return gt

    def box(self, s, left, top, width, height, fill=LIGHT, line=None):
        b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        b.fill.solid(); b.fill.fore_color.rgb = fill
        if line is None:
            b.line.fill.background()
        else:
            b.line.color.rgb = line
        b.adjustments[0] = 0.08
        return b
