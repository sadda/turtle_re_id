import colorsys
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
WIDTH = 960
HEIGHT = 540
FPS = 24
CRF = 32
FADE = 0.5
OUTPUT = "kiki_{lang}.mp4"

BRAND = "Turtlewatch Egypt 2.0"
LOGO = "logo.png"
SUBMISSION_URL = "https://turtlewatchegypt.net/share-marine-turtles-photos/"
SUBMISSION_LABEL = "turtlewatchegypt.net"

BACKGROUND = (12, 30, 42)
TEXT_COLOR = (255, 255, 255)
ACCENT = (255, 196, 60)
LABEL_TO_PART = {
    "head": "head",
    "flipper_fl": "front flipper",
    "flipper_fr": "front flipper",
    "flipper_hl": "hind flipper",
    "flipper_hr": "hind flipper",
}
PART_COLORS = {
    "head": (255, 196, 60),
    "flipper_fl": (80, 220, 255),
    "flipper_fr": (80, 220, 255),
    "flipper_hl": (255, 110, 200),
    "flipper_hr": (255, 110, 200),
}

# ---------------------------------------------------------------------------
# Storyboard: edit texts, durations and order here
# ---------------------------------------------------------------------------
STORYBOARD = [
    {"type": "intro", "duration": 5.0, "id": "intro"},
    {"type": "fingerprint", "duration": 5.0, "id": "fingerprint"},
    {"type": "photo", "duration": 3.5, "photo": "2019", "id": "photo1", "fade_out": False},
    {"type": "detect", "duration": 4.5, "photo": "2019", "id": "detect1", "fade_in": False},
    {"type": "photo", "duration": 3.5, "photo": "2023", "id": "photo2", "fade_out": False},
    {"type": "detect", "duration": 4.0, "photo": "2023", "id": "detect2", "fade_in": False},
    {"type": "match", "duration": 7.0, "part": "head", "id": "match_head"},
    {"type": "match", "duration": 7.0, "part": "front flipper", "id": "match_front"},
    {"type": "match", "duration": 7.0, "part": "hind flipper", "id": "match_hind"},
    {"type": "summary", "duration": 5.0, "id": "summary"},
    {"type": "outro", "duration": 7.0, "id": "outro"},
]

# ---------------------------------------------------------------------------
# Languages: edit texts here
# ---------------------------------------------------------------------------
LANGUAGES = {
    "en": {
        "font_bold": "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "font_regular": "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "direction": "ltr",
        "parts": {"head": "Head", "front flipper": "Front flipper", "hind flipper": "Hind flipper"},
        "texts": {
            "intro": {"title": "Meet Kiki", "text": "A green sea turtle from Marsa Abu Dabbab, Egypt"},
            "fingerprint": {"title": "Every turtle is unique", "text": "The scales on its head and flippers form a pattern, like a fingerprint"},
            "photo1": {"text": "October 2019: a diver takes a photo of Kiki"},
            "detect1": {"text": "AI finds the head and the flippers"},
            "photo2": {"text": "February 2023: more than 3 years later..."},
            "detect2": {"text": "Is this the same turtle?"},
            "match_head": {"title": "Comparing the patterns", "text": "Head: {n} matching points", "note": "Even a photo from above can be matched with a photo from the side."},
            "match_front": {"title": "Comparing the patterns", "text": "Front flipper: {n} matching points", "note": "For every match, the AI also tells us how confident it is."},
            "match_hind": {"title": "Comparing the patterns", "text": "Hind flipper: {n} matching points", "note": "We are the first to identify turtles by their hind flippers."},
            "summary": {"title": "It's Kiki again!", "text": "All three body parts match"},
            "outro": {"title": "Your photos help", "text": "We follow turtles for years, without ever touching them.\nShare your turtle photos with us!", "qr": "Scan to share your photos"},
        },
    },
    "de": {
        "font_bold": "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "font_regular": "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "direction": "ltr",
        "parts": {"head": "Kopf", "front flipper": "Vorderflosse", "hind flipper": "Hinterflosse"},
        "texts": {
            "intro": {"title": "Das ist Kiki", "text": "Eine Grüne Meeresschildkröte aus Marsa Abu Dabbab, Ägypten"},
            "fingerprint": {"title": "Jede Schildkröte ist einzigartig", "text": "Die Schuppen an Kopf und Flossen bilden ein Muster, wie ein Fingerabdruck"},
            "photo1": {"text": "Oktober 2019: Ein Taucher fotografiert Kiki"},
            "detect1": {"text": "KI erkennt den Kopf und die Flossen"},
            "photo2": {"text": "Februar 2023: mehr als 3 Jahre später ..."},
            "detect2": {"text": "Ist das dieselbe Schildkröte?"},
            "match_head": {"title": "Wir vergleichen die Muster", "text": "Kopf: {n} übereinstimmende Punkte", "note": "Sogar ein Foto von oben lässt sich mit einem Foto von der Seite vergleichen."},
            "match_front": {"title": "Wir vergleichen die Muster", "text": "Vorderflosse: {n} übereinstimmende Punkte", "note": "Bei jedem Treffer gibt die KI auch an, wie sicher sie sich ist."},
            "match_hind": {"title": "Wir vergleichen die Muster", "text": "Hinterflosse: {n} übereinstimmende Punkte", "note": "Wir sind die Ersten, die Schildkröten an ihren Hinterflossen erkennen."},
            "summary": {"title": "Es ist wieder Kiki!", "text": "Alle drei Körperteile stimmen überein"},
            "outro": {"title": "Ihre Fotos helfen", "text": "Wir begleiten Schildkröten über Jahre, ohne sie je zu berühren.\nTeilen Sie Ihre Schildkrötenfotos mit uns!", "qr": "Scannen und Fotos teilen"},
        },
    },
    "ar": {
        "font_bold": "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
        "font_regular": "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "direction": "rtl",
        "parts": {"head": "الرأس", "front flipper": "الزعنفة الأمامية", "hind flipper": "الزعنفة الخلفية"},
        "texts": {
            "intro": {"title": "تعرّفوا على كيكي", "text": "سلحفاة بحرية خضراء من مرسى أبو دباب، مصر"},
            "fingerprint": {"title": "كل سلحفاة فريدة من نوعها", "text": "تشكّل الحراشف على رأسها وزعانفها نمطًا مميزًا، مثل بصمة الإصبع"},
            "photo1": {"text": "أكتوبر 2019: غوّاص يلتقط صورة لكيكي"},
            "detect1": {"text": "الذكاء الاصطناعي يحدّد الرأس والزعانف"},
            "photo2": {"text": "فبراير 2023: بعد أكثر من 3 سنوات..."},
            "detect2": {"text": "هل هذه السلحفاة نفسها؟"},
            "match_head": {"title": "مقارنة الأنماط", "text": "الرأس: عدد النقاط المتطابقة {n}", "note": "يمكن أيضًا مطابقة صورة من الأعلى مع صورة جانبية."},
            "match_front": {"title": "مقارنة الأنماط", "text": "الزعنفة الأمامية: عدد النقاط المتطابقة {n}", "note": "مع كل مطابقة، يخبرنا الذكاء الاصطناعي أيضًا بمدى ثقته."},
            "match_hind": {"title": "مقارنة الأنماط", "text": "الزعنفة الخلفية: عدد النقاط المتطابقة {n}", "note": "نحن أول من يتعرّف على السلاحف من خلال زعانفها الخلفية."},
            "summary": {"title": "إنها كيكي مجددًا!", "text": "الأجزاء الثلاثة كلها متطابقة"},
            "outro": {"title": "صوركم تساعدنا", "text": "نتابع السلاحف لسنوات دون أن نلمسها أبدًا.\nشاركونا صوركم للسلاحف!", "qr": "امسحوا الرمز لمشاركة صوركم"},
        },
    },
}
LANG = LANGUAGES["en"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_ASSETS = os.path.join(ROOT, "assets")

with open(os.path.join(ROOT_ASSETS, "assets.json")) as file:
    ASSETS = json.load(file)
PHOTOS = {key: Image.open(os.path.join(ROOT_ASSETS, name)).convert("RGB") for key, name in ASSETS["photos"].items()}
MATCHES = {m["part"]: m for m in ASSETS["matches"]}
CROPS = {name: Image.open(os.path.join(ROOT_ASSETS, name)).convert("RGB") for m in ASSETS["matches"] for name in [m["crop1"], m["crop2"]]}
YEARS = list(ASSETS["photos"])
LOGO_PATH = os.path.join(ROOT_ASSETS, LOGO)
LOGO_IMAGE = Image.open(LOGO_PATH).convert("RGBA") if os.path.exists(LOGO_PATH) else None


def make_qr(url: str) -> Image.Image:
    qr = cv2.QRCodeEncoder.create().encode(url)
    return Image.fromarray(qr).convert("RGB")


QR_IMAGE = make_qr(SUBMISSION_URL)


def font(size_rel: float, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(LANG["font_bold"] if bold else LANG["font_regular"], round(HEIGHT * size_rel))


def text(scene: dict, key: str) -> str:
    return LANG["texts"][scene["id"]][key]


def part_name(label: str) -> str:
    return LANG["parts"][LABEL_TO_PART[label]]


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def progress(t: float, start: float, end: float) -> float:
    return ease((t - start) / (end - start))


def new_frame() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=fnt, direction=LANG["direction"]) <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_text_block(img: Image.Image, text: str, fnt: ImageFont.FreeTypeFont, center_y: float, color: tuple, alpha: float = 1.0) -> None:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    lines = wrap(draw, text, fnt, WIDTH * 0.85)
    line_height = fnt.size * 1.3
    y = center_y - line_height * len(lines) / 2
    for line in lines:
        w = draw.textlength(line, font=fnt, direction=LANG["direction"])
        draw.text(((WIDTH - w) / 2, y), line, font=fnt, fill=color + (round(255 * alpha),), direction=LANG["direction"])
        y += line_height
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


def draw_caption(img: Image.Image, text: str, alpha: float = 1.0, note: str | None = None, note_alpha: float = 1.0) -> None:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    blocks = [(font(0.05), wrap(draw, text, font(0.05), WIDTH * 0.85), alpha)]
    if note is not None:
        blocks.append((font(0.04, bold=False), wrap(draw, note, font(0.04, bold=False), WIDTH * 0.85), alpha * note_alpha))
    band_height = sum(fnt.size * 1.3 * len(lines) for fnt, lines, _ in blocks) + HEIGHT * 0.05
    draw.rectangle([0, HEIGHT - band_height, WIDTH, HEIGHT], fill=(0, 0, 0, round(160 * alpha)))
    y = HEIGHT - band_height + HEIGHT * 0.025
    for fnt, lines, a in blocks:
        for line in lines:
            w = draw.textlength(line, font=fnt, direction=LANG["direction"])
            draw.text(((WIDTH - w) / 2, y), line, font=fnt, fill=TEXT_COLOR + (round(255 * a),), direction=LANG["direction"])
            y += fnt.size * 1.3
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


def draw_title(img: Image.Image, text: str, alpha: float = 1.0) -> None:
    draw_text_block(img, text, font(0.065), HEIGHT * 0.08, ACCENT, alpha)


def place_photo(img: Image.Image, zoom: float) -> tuple[Image.Image, float, float, float]:
    scale = min(WIDTH / img.width, HEIGHT / img.height) * zoom
    w = round(img.width * scale)
    h = round(img.height * scale)
    x = (WIDTH - w) / 2
    y = (HEIGHT - h) / 2
    return img.resize((w, h), Image.BILINEAR), scale, x, y


def fit(img: Image.Image, max_w: float, max_h: float) -> tuple[Image.Image, float]:
    scale = min(max_w / img.width, max_h / img.height)
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS), scale


def match_colors(n: int) -> list[tuple]:
    colors = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.65, 1.0)
        colors.append((round(r * 255), round(g * 255), round(b * 255)))
    return colors


def draw_checkmark(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, p: float) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(60, 180, 90))
    pts = [(cx - r * 0.5, cy), (cx - r * 0.1, cy + r * 0.4), (cx + r * 0.55, cy - r * 0.4)]
    if p <= 0.5:
        q = p / 0.5
        end = (pts[0][0] + (pts[1][0] - pts[0][0]) * q, pts[0][1] + (pts[1][1] - pts[0][1]) * q)
        draw.line([pts[0], end], fill=(255, 255, 255), width=max(2, round(r * 0.25)))
    else:
        q = (p - 0.5) / 0.5
        end = (pts[1][0] + (pts[2][0] - pts[1][0]) * q, pts[1][1] + (pts[2][1] - pts[1][1]) * q)
        draw.line([pts[0], pts[1], end], fill=(255, 255, 255), width=max(2, round(r * 0.25)), joint="curve")


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------
def scene_title(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    a1 = progress(t, 0.2, 1.0)
    a2 = progress(t, 0.9, 1.8)
    draw_text_block(img, text(scene, "title"), font(0.11), HEIGHT * 0.38, ACCENT, a1)
    draw_text_block(img, text(scene, "text"), font(0.05, bold=False), HEIGHT * 0.62, TEXT_COLOR, a2)
    return img


def scene_intro(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    a0 = progress(t, 0.1, 0.9)
    a1 = progress(t, 1.0, 1.8)
    a2 = progress(t, 1.7, 2.5)
    brand_y = HEIGHT * 0.22
    if LOGO_IMAGE is not None:
        logo, _ = fit(LOGO_IMAGE, WIDTH * 0.4, HEIGHT * 0.22)
        logo.putalpha(logo.getchannel("A").point(lambda v: round(v * a0)))
        img.paste(logo, (round((WIDTH - logo.width) / 2), round(HEIGHT * 0.17 - logo.height / 2)), logo)
        brand_y = HEIGHT * 0.36
    draw_text_block(img, BRAND, ImageFont.truetype(LANGUAGES["en"]["font_bold"], round(HEIGHT * 0.06)), brand_y, TEXT_COLOR, a0)
    draw_text_block(img, text(scene, "title"), font(0.1), HEIGHT * 0.56, ACCENT, a1)
    draw_text_block(img, text(scene, "text"), font(0.05, bold=False), HEIGHT * 0.74, TEXT_COLOR, a2)
    return img


def scene_outro(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    a1 = progress(t, 0.2, 1.0)
    a2 = progress(t, 0.9, 1.7)
    a3 = progress(t, 1.6, 2.4)
    draw_text_block(img, text(scene, "title"), font(0.08), HEIGHT * 0.12, ACCENT, a1)
    draw_text_block(img, text(scene, "text"), font(0.045, bold=False), HEIGHT * 0.3, TEXT_COLOR, a2)
    size = round(HEIGHT * 0.38)
    qr = QR_IMAGE.resize((size, size), Image.NEAREST)
    qr = Image.blend(new_frame().crop((0, 0, size, size)), qr, a3)
    img.paste(qr, (round((WIDTH - size) / 2), round(HEIGHT * 0.45)))
    label_y = HEIGHT * 0.45 + size + HEIGHT * 0.05
    draw_text_block(img, text(scene, "qr"), font(0.035), label_y, TEXT_COLOR, a3)
    url_font = ImageFont.truetype(LANGUAGES["en"]["font_bold"], round(HEIGHT * 0.035))
    draw_text_block(img, SUBMISSION_LABEL, url_font, label_y + HEIGHT * 0.055, ACCENT, a3)
    return img


def scene_fingerprint(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    match = MATCHES["head"]
    crop, _ = fit(CROPS[match["crop1"]], WIDTH * 0.5, HEIGHT * 0.55)
    zoom = 1.0 + 0.08 * (t / scene["duration"])
    crop = crop.resize((round(crop.width * zoom), round(crop.height * zoom)), Image.BILINEAR)
    img.paste(crop, (round((WIDTH - crop.width) / 2), round(HEIGHT * 0.48 - crop.height / 2)))
    draw_title(img, text(scene, "title"), progress(t, 0.2, 1.0))
    draw_caption(img, text(scene, "text"), progress(t, 1.0, 1.8))
    return img


def scene_photo(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    zoom = 1.0 + 0.05 * ease(t / scene["duration"])
    photo, _, x, y = place_photo(PHOTOS[scene["photo"]], zoom)
    img.paste(photo, (round(x), round(y)))
    draw_caption(img, text(scene, "text"), progress(t, 0.3, 1.0))
    return img


def scene_detect(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    photo, scale, ox, oy = place_photo(PHOTOS[scene["photo"]], 1.05)
    img.paste(photo, (round(ox), round(oy)))

    detections = sorted(ASSETS["detections"][scene["photo"]], key=lambda d: list(LABEL_TO_PART).index(d["label"]))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fnt = font(0.035)
    step = min(0.6, (scene["duration"] - 1.5) / max(1, len(detections)))
    for i, det in enumerate(detections):
        p = progress(t, 0.3 + i * step, 0.3 + i * step + 0.4)
        if p <= 0:
            continue
        x, y, w, h = det["bbox"]
        cx = ox + (x + w / 2) * scale
        cy = oy + (y + h / 2) * scale
        hw = w * scale / 2 * (0.6 + 0.4 * p)
        hh = h * scale / 2 * (0.6 + 0.4 * p)
        color = PART_COLORS[det["label"]]
        polygon = [(ox + px * scale, oy + py * scale) for px, py in det["polygon"]]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=color + (round(110 * p),))
        draw.rectangle([cx - hw, cy - hh, cx + hw, cy + hh], outline=color + (round(255 * p),), width=3)
        name = part_name(det["label"])
        tw = draw.textlength(name, font=fnt, direction=LANG["direction"])
        tx = cx - hw
        ty = cy - hh - fnt.size * 1.4
        draw.rectangle([tx, ty, tx + tw + fnt.size * 0.6, ty + fnt.size * 1.4], fill=color + (round(230 * p),))
        draw.text((tx + fnt.size * 0.3, ty + fnt.size * 0.1), name, font=fnt, fill=(0, 0, 0, round(255 * p)), direction=LANG["direction"])
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw_caption(img, text(scene, "text"), 1.0)
    return img


def scene_match(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    match = MATCHES[scene["part"]]
    gap = WIDTH * 0.04
    max_w = (WIDTH * 0.9 - gap) / 2
    max_h = HEIGHT * 0.5
    crop1, s1 = fit(CROPS[match["crop1"]], max_w, max_h)
    crop2, s2 = fit(CROPS[match["crop2"]], max_w, max_h)
    center_y = HEIGHT * 0.45
    x1 = WIDTH / 2 - gap / 2 - crop1.width
    y1 = center_y - crop1.height / 2
    x2 = WIDTH / 2 + gap / 2
    y2 = center_y - crop2.height / 2
    img.paste(crop1, (round(x1), round(y1)))
    img.paste(crop2, (round(x2), round(y2)))

    draw = ImageDraw.Draw(img)
    year_fnt = font(0.04)
    for year, x, w, y in [(YEARS[0], x1, crop1.width, y1), (YEARS[1], x2, crop2.width, y2)]:
        tw = draw.textlength(year, font=year_fnt)
        draw.text((x + (w - tw) / 2, y - year_fnt.size * 1.5), year, font=year_fnt, fill=TEXT_COLOR)

    kpts1 = np.asarray(match["kpts1"])
    kpts2 = np.asarray(match["kpts2"])
    n = len(kpts1)
    order = np.argsort(kpts1[:, 1])
    colors = match_colors(n)
    start = 0.8
    end = scene["duration"] - 3.0
    r = max(2, round(HEIGHT * 0.005))
    n_drawn = 0
    for rank, k in enumerate(order):
        t0 = start + (end - start) * rank / max(1, n)
        p = progress(t, t0, t0 + 0.5)
        if p <= 0:
            continue
        n_drawn += 1
        ax = x1 + kpts1[k, 0] * s1
        ay = y1 + kpts1[k, 1] * s1
        bx = x2 + kpts2[k, 0] * s2
        by = y2 + kpts2[k, 1] * s2
        ex = ax + (bx - ax) * p
        ey = ay + (by - ay) * p
        draw.line([(ax, ay), (ex, ey)], fill=colors[rank], width=2)
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], outline=colors[rank], width=2)
        if p >= 1:
            draw.ellipse([bx - r, by - r, bx + r, by + r], outline=colors[rank], width=2)

    draw_title(img, text(scene, "title"), 1.0)
    note_alpha = progress(t, end - 0.5, end + 0.3)
    draw_caption(img, text(scene, "text").format(n=n_drawn), 1.0, note=text(scene, "note"), note_alpha=note_alpha)
    return img


def scene_summary(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    draw = ImageDraw.Draw(img)
    parts = list(MATCHES)
    col_w = WIDTH * 0.9 / len(parts)
    pair_h = HEIGHT * 0.22
    name_fnt = font(0.04)
    for i, part in enumerate(parts):
        match = MATCHES[part]
        cx = WIDTH * 0.05 + col_w * (i + 0.5)
        p = progress(t, 0.3 + i * 0.6, 0.9 + i * 0.6)
        if p <= 0:
            continue
        crop1, _ = fit(CROPS[match["crop1"]], col_w * 0.42, pair_h)
        crop2, _ = fit(CROPS[match["crop2"]], col_w * 0.42, pair_h)
        y = HEIGHT * 0.42
        img.paste(Image.blend(new_frame().crop((0, 0) + crop1.size), crop1, p), (round(cx - col_w * 0.02 - crop1.width), round(y - crop1.height / 2)))
        img.paste(Image.blend(new_frame().crop((0, 0) + crop2.size), crop2, p), (round(cx + col_w * 0.02), round(y - crop2.height / 2)))
        name = part_name(match["label1"])
        tw = draw.textlength(name, font=name_fnt, direction=LANG["direction"])
        draw.text((cx - tw / 2, y + pair_h / 2 + HEIGHT * 0.03), name, font=name_fnt, fill=TEXT_COLOR, direction=LANG["direction"])
        q = progress(t, 0.9 + i * 0.6, 1.4 + i * 0.6)
        if q > 0:
            draw_checkmark(draw, cx, y + pair_h / 2 + HEIGHT * 0.15, HEIGHT * 0.035, q)
    draw_title(img, text(scene, "title"), progress(t, 2.2, 3.0))
    draw_caption(img, text(scene, "text"), progress(t, 2.2, 3.0))
    return img


SCENES = {
    "intro": scene_intro,
    "outro": scene_outro,
    "title": scene_title,
    "fingerprint": scene_fingerprint,
    "photo": scene_photo,
    "detect": scene_detect,
    "match": scene_match,
    "summary": scene_summary,
}

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def apply_fade(img: Image.Image, scene: dict, t: float) -> Image.Image:
    alpha = 1.0
    if scene.get("fade_in", True):
        alpha = min(alpha, t / FADE)
    if scene.get("fade_out", True):
        alpha = min(alpha, (scene["duration"] - t) / FADE)
    alpha = min(max(alpha, 0.0), 1.0)
    if alpha >= 1:
        return img
    return Image.blend(new_frame(), img, alpha)


def render(output: str) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for scene in STORYBOARD:
        scene_fn = SCENES[scene["type"]]
        n_frames = round(scene["duration"] * FPS)
        for i in range(n_frames):
            t = i / FPS
            frame = scene_fn(scene, t)
            frame = apply_fade(frame, scene, t)
            process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()


if __name__ == "__main__":
    langs = sys.argv[1:] or list(LANGUAGES)
    for lang in langs:
        LANG = LANGUAGES[lang]
        render(os.path.join(ROOT, OUTPUT.format(lang=lang)))
