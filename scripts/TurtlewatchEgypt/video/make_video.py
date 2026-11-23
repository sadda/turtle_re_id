import colorsys
import json
import os
import argparse
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
WIDTH = 1920
HEIGHT = 1080
FPS = 24
CRF = 20
FADE = 0.5
OUTPUT = "kiki_{lang}_{width}x{height}.mp4"

LOGO = "logo.png"
SUBMISSION_URL = "https://turtlewatchegypt.net/share-marine-turtles-photos/"
SUBMISSION_LABEL = "turtlewatchegypt.net"

ROOT_PRESENTATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "presentation")
SLIDE_FIRST = "slide_first.png"
SLIDE_LAST = "slide_last.png"
BACKGROUND_IMAGE = "image25.png"
END_CARD_IMAGE = "image25.png"
END_CARD_CREDIT = "© Micol Montagna"
BACKGROUND_ANCHOR_X = 0.42
BACKGROUND_BLUR = 0.03
BACKGROUND_DARKEN = 0.35
PHOTO_BLUR = 0.04
CORNER_LOGO = "image13.png"
CORNER_LOGO_HEIGHT = 0.036
BOX_ALPHA = 0.5
BOX_MAX_WIDTH = 0.7
TITLE_SPACING = 0.08
PORTRAIT_SAFE_TOP = 0.13
PORTRAIT_SAFE_BOTTOM = 0.24
PORTRAIT_TEXT_SCALE = 1.15
PORTRAIT_HEADER_SHIFT = 0.10
PORTRAIT_CAPTION_SHIFT = 0.16
PORTRAIT_CONTENT_SHIFT = 0.10
PORTRAIT_PARTS = ["head", "hind flipper"]
PORTRAIT_CROP_MARGIN = 0.12
PORTRAIT_CROP = {"head": "", "hind flipper": "y"}
QUICKSAND_BOLD = "/usr/share/fonts/truetype/quicksand/Quicksand-Bold.ttf"
QUICKSAND_REGULAR = "/usr/share/fonts/truetype/quicksand/Quicksand-Regular.ttf"

TEXT_COLOR = (255, 255, 255)
ACCENT = (255, 255, 255)
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
# Storyboards: edit durations and order here (landscape and portrait/reels)
# ---------------------------------------------------------------------------
STORYBOARD_LANDSCAPE = [
    {"type": "slide", "duration": 4.0, "image": SLIDE_FIRST, "fade_in": False},
    {"type": "intro", "duration": 5.0, "id": "intro"},
    {"type": "fingerprint", "duration": 5.0, "id": "fingerprint"},
    {"type": "detect", "duration": 7.5, "photo": "2019", "id": "detect1", "photo_id": "photo1", "boxes_at": 3.0},
    {"type": "detect", "duration": 7.0, "photo": "2023", "id": "detect2", "photo_id": "photo2", "boxes_at": 3.0},
    {"type": "match", "duration": 7.0, "part": "hind flipper", "id": "match_hind"},
    {"type": "match", "duration": 7.0, "part": "front flipper", "id": "match_front"},
    {"type": "match", "duration": 7.0, "part": "head", "id": "match_head"},
    {"type": "summary", "duration": 5.0, "id": "summary"},
    {"type": "outro", "duration": 7.0, "id": "outro"},
    {"type": "slide", "duration": 5.0, "image": SLIDE_LAST, "fade_out": False},
]

STORYBOARD_PORTRAIT = [
    {"type": "hook", "duration": 3.5, "id": "hook", "fade_in": False},
    {"type": "intro", "duration": 3.5, "id": "intro"},
    {"type": "detect", "duration": 5.5, "photo": "2019", "id": "detect1", "photo_id": "photo1", "boxes_at": 2.2},
    {"type": "detect", "duration": 5.0, "photo": "2023", "id": "detect2", "photo_id": "photo2", "boxes_at": 2.0},
    {"type": "match", "duration": 5.5, "part": "hind flipper", "id": "match_hind", "layout": "column"},
    {"type": "match", "duration": 5.5, "part": "head", "id": "match_head", "layout": "column"},
    {"type": "summary", "duration": 4.0, "id": "summary_reel", "parts": PORTRAIT_PARTS},
    {"type": "outro", "duration": 5.0, "id": "outro"},
    {"type": "endcard", "duration": 3.5, "id": "endcard", "fade_out": False},
]

# ---------------------------------------------------------------------------
# Languages: edit texts here
# ---------------------------------------------------------------------------
LANGUAGES = {
    "en": {
        "font_bold": QUICKSAND_BOLD,
        "font_regular": QUICKSAND_REGULAR,
        "direction": "ltr",
        "parts": {"head": "Head", "front flipper": "Front flipper", "hind flipper": "Hind flipper"},
        "texts": {
            "hook": {"title": "Same turtle?", "text": "2019 vs 2023"},
            "endcard": {"title": "Thank you!"},
            "intro": {"title": "Meet Kiki", "text": "A green turtle from Abu Dabbab, Egypt"},
            "fingerprint": {"title": "Every turtle is unique", "text": "Its scales are unique, like a fingerprint"},
            "photo1": {"text": "October 2019: a diver takes a photo of Kiki"},
            "detect1": {"text": "AI finds the head and the flippers", "title": "Finding body parts"},
            "photo2": {"text": "February 2023: more than 3 years later..."},
            "detect2": {"text": "Is this the same turtle?", "title": "Finding body parts"},
            "match_head": {"title": "Comparing the patterns", "text": "Head: {n} matching points", "note": "Top and side photos match, even with little overlap."},
            "match_front": {"title": "Comparing the patterns", "text": "Front flipper: {n} matching points", "note": "The AI also tells us how sure it is about its answer."},
            "match_hind": {"title": "Comparing the patterns", "text": "Hind flipper: {n} matching points", "note": "We are the first to identify turtles by their hind flippers."},
            "summary": {"title": "It's Kiki again!", "text": "All three body parts match"},
            "summary_reel": {"title": "It's Kiki again!", "text": "All body parts match"},
            "outro": {"title": "Your photos help", "text": "We follow turtles for years, without touching them.\nShare your turtle photos!", "qr": "Scan to share your photos"},
        },
    },
    "de": {
        "font_bold": QUICKSAND_BOLD,
        "font_regular": QUICKSAND_REGULAR,
        "direction": "ltr",
        "parts": {"head": "Kopf", "front flipper": "Vorderflosse", "hind flipper": "Hinterflosse"},
        "texts": {
            "hook": {"title": "Dieselbe Schildkröte?", "text": "2019 vs. 2023"},
            "endcard": {"title": "Danke!"},
            "intro": {"title": "Das ist Kiki", "text": "Grüne Schildkröte aus Abu Dabbab, Ägypten"},
            "fingerprint": {"title": "Jede ist einzigartig", "text": "Ihre Schuppen sind wie ein Fingerabdruck"},
            "photo1": {"text": "Oktober 2019: Ein Taucher fotografiert Kiki"},
            "detect1": {"text": "KI findet Kopf und Flossen", "title": "Körperteile erkennen"},
            "photo2": {"text": "Februar 2023: mehr als 3 Jahre später ..."},
            "detect2": {"text": "Ist das dieselbe Schildkröte?", "title": "Körperteile erkennen"},
            "match_head": {"title": "Muster vergleichen", "text": "Kopf: {n} Treffer", "note": "Fotos von oben und von der Seite passen zusammen."},
            "match_front": {"title": "Muster vergleichen", "text": "Vorderflosse: {n} Treffer", "note": "Die KI sagt auch, wie sicher sie sich ist."},
            "match_hind": {"title": "Muster vergleichen", "text": "Hinterflosse: {n} Treffer", "note": "Als Erste erkennen wir sie an den Hinterflossen."},
            "summary": {"title": "Es ist wieder Kiki!", "text": "Alle drei Körperteile stimmen überein"},
            "summary_reel": {"title": "Es ist wieder Kiki!", "text": "Alle Körperteile stimmen überein"},
            "outro": {"title": "Ihre Fotos helfen", "text": "Jahrelange Beobachtung, ganz ohne Berührung.\nTeilen Sie Ihre Fotos mit uns!", "qr": "Scannen und Fotos teilen"},
        },
    },
    "it": {
        "font_bold": QUICKSAND_BOLD,
        "font_regular": QUICKSAND_REGULAR,
        "direction": "ltr",
        "parts": {"head": "Testa", "front flipper": "Pinna anteriore", "hind flipper": "Pinna posteriore"},
        "texts": {
            "hook": {"title": "Stessa tartaruga?", "text": "2019 vs 2023"},
            "endcard": {"title": "Grazie!"},
            "intro": {"title": "Ecco Kiki", "text": "Una tartaruga verde di Abu Dabbab, Egitto"},
            "fingerprint": {"title": "Ogni tartaruga è unica", "text": "Squame uniche, come un'impronta digitale"},
            "photo1": {"text": "Ottobre 2019: un sub fotografa Kiki"},
            "detect1": {"text": "L'IA trova la testa e le pinne", "title": "Riconoscere il corpo"},
            "photo2": {"text": "Febbraio 2023: più di 3 anni dopo..."},
            "detect2": {"text": "È la stessa tartaruga?", "title": "Riconoscere il corpo"},
            "match_head": {"title": "Confrontiamo i disegni", "text": "Testa: {n} punti uguali", "note": "Una foto dall'alto e una di lato possono corrispondere."},
            "match_front": {"title": "Confrontiamo i disegni", "text": "Pinna anteriore: {n} punti uguali", "note": "L'IA ci dice anche quanto è sicura della sua risposta."},
            "match_hind": {"title": "Confrontiamo i disegni", "text": "Pinna posteriore: {n} punti uguali", "note": "Siamo i primi a riconoscerle dalle pinne posteriori."},
            "summary": {"title": "È di nuovo Kiki!", "text": "Tutte e tre le parti corrispondono"},
            "summary_reel": {"title": "È di nuovo Kiki!", "text": "Tutte le parti corrispondono"},
            "outro": {"title": "Le tue foto aiutano", "text": "Seguiamo le tartarughe per anni, senza toccarle.\nCondividi le tue foto!", "qr": "Scansiona per condividere le foto"},
        },
    },
    "cs": {
        "font_bold": QUICKSAND_BOLD,
        "font_regular": QUICKSAND_REGULAR,
        "direction": "ltr",
        "parts": {"head": "Hlava", "front flipper": "Přední ploutev", "hind flipper": "Zadní ploutev"},
        "texts": {
            "hook": {"title": "Stejná želva?", "text": "2019 vs 2023"},
            "endcard": {"title": "Děkujeme!"},
            "intro": {"title": "Toto je Kiki", "text": "Kareta obrovská z Abu Dabbab, Egypt"},
            "fingerprint": {"title": "Každá želva je jiná", "text": "Její šupiny jsou jako otisk prstu"},
            "photo1": {"text": "Říjen 2019: potápěč fotí Kiki"},
            "detect1": {"text": "AI najde hlavu a ploutve", "title": "Hledání částí těla"},
            "photo2": {"text": "Únor 2023: o více než 3 roky později..."},
            "detect2": {"text": "Je to stejná želva?", "title": "Hledání částí těla"},
            "match_head": {"title": "Porovnání vzorů", "text": "Hlava – shodné body: {n}", "note": "Fotky shora a z boku se shodují i při malém překryvu."},
            "match_front": {"title": "Porovnání vzorů", "text": "Přední ploutev – shodné body: {n}", "note": "AI nám také řekne, jak si je jistá."},
            "match_hind": {"title": "Porovnání vzorů", "text": "Zadní ploutev – shodné body: {n}", "note": "Jako první poznáváme želvy podle zadních ploutví."},
            "summary": {"title": "Je to zase Kiki!", "text": "Všechny tři části těla se shodují"},
            "summary_reel": {"title": "Je to zase Kiki!", "text": "Všechny části těla se shodují"},
            "outro": {"title": "Fotky pomáhají", "text": "Sledujeme želvy roky, aniž bychom se jich dotkli.\nSdílejte své fotky želv!", "qr": "Naskenujte a sdílejte fotky"},
        },
    },
    "ar": {
        "font_bold": "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
        "font_regular": "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "direction": "rtl",
        "parts": {"head": "الرأس", "front flipper": "الزعنفة الأمامية", "hind flipper": "الزعنفة الخلفية"},
        "texts": {
            "hook": {"title": "السلحفاة نفسها؟", "text": "2019 مقابل 2023"},
            "endcard": {"title": "شكرًا!"},
            "intro": {"title": "تعرّفوا على كيكي", "text": "سلحفاة بحرية خضراء من مرسى أبو دباب، مصر"},
            "fingerprint": {"title": "كل سلحفاة فريدة من نوعها", "text": "حراشف الرأس والزعانف مثل بصمة الإصبع"},
            "photo1": {"text": "أكتوبر 2019: غوّاص يلتقط صورة لكيكي"},
            "detect1": {"text": "الذكاء الاصطناعي يحدّد الرأس والزعانف", "title": "تحديد أجزاء الجسم"},
            "photo2": {"text": "فبراير 2023: بعد أكثر من 3 سنوات..."},
            "detect2": {"text": "هل هذه السلحفاة نفسها؟", "title": "تحديد أجزاء الجسم"},
            "match_head": {"title": "مقارنة الأنماط", "text": "الرأس: عدد النقاط المتطابقة {n}", "note": "يمكن مطابقة صورة علوية بصورة جانبية، حتى مع تداخل قليل."},
            "match_front": {"title": "مقارنة الأنماط", "text": "الزعنفة الأمامية: عدد النقاط المتطابقة {n}", "note": "ويخبرنا الذكاء الاصطناعي أيضًا بمدى ثقته في إجابته."},
            "match_hind": {"title": "مقارنة الأنماط", "text": "الزعنفة الخلفية: عدد النقاط المتطابقة {n}", "note": "نحن أول من يتعرّف على السلاحف من خلال زعانفها الخلفية."},
            "summary": {"title": "إنها كيكي مجددًا!", "text": "الأجزاء الثلاثة كلها متطابقة"},
            "summary_reel": {"title": "إنها كيكي مجددًا!", "text": "جميع أجزاء الجسم متطابقة"},
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
LOGO_PATH = os.path.join(ROOT, LOGO)
LOGO_IMAGE = Image.open(LOGO_PATH).convert("RGBA")


def make_qr(url: str) -> Image.Image:
    qr = cv2.QRCodeEncoder.create().encode(url)
    return Image.fromarray(qr).convert("RGB")


QR_IMAGE = make_qr(SUBMISSION_URL)


def unit() -> int:
    return min(WIDTH, HEIGHT)


def is_portrait() -> bool:
    return HEIGHT > WIDTH


def safe_top() -> float:
    return HEIGHT * PORTRAIT_SAFE_TOP if is_portrait() else 0.0


def safe_bottom() -> float:
    return HEIGHT * (1 - PORTRAIT_SAFE_BOTTOM) if is_portrait() else HEIGHT


def font(size_rel: float, bold: bool = True) -> ImageFont.FreeTypeFont:
    scale = PORTRAIT_TEXT_SCALE if is_portrait() else 1.0
    return ImageFont.truetype(LANG["font_bold"] if bold else LANG["font_regular"], round(unit() * size_rel * scale))


def text(scene: dict, key: str) -> str:
    return LANG["texts"][scene["id"]][key]


def part_name(label: str) -> str:
    return LANG["parts"][LABEL_TO_PART[label]]


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def progress(t: float, start: float, end: float) -> float:
    return ease((t - start) / (end - start))


CACHE = {}
CURRENT_BACKGROUND = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))


def cover(img: Image.Image, anchor_x: float = 0.5) -> Image.Image:
    scale = max(WIDTH / img.width, HEIGHT / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x = round((img.width - WIDTH) * anchor_x)
    y = (img.height - HEIGHT) // 2
    return img.crop((x, y, x + WIDTH, y + HEIGHT))


def blurred_background(key: str, img: Image.Image, blur: float, darken: float, anchor_x: float = 0.5) -> Image.Image:
    cache_key = (key, WIDTH, HEIGHT)
    if cache_key not in CACHE:
        background = cover(img.convert("RGB"), anchor_x).filter(ImageFilter.GaussianBlur(unit() * blur))
        background = Image.blend(background, Image.new("RGB", background.size, (0, 0, 0)), darken)
        CACHE[cache_key] = background
    return CACHE[cache_key]


def presentation_image(name: str) -> Image.Image:
    if name not in CACHE:
        CACHE[name] = Image.open(os.path.join(ROOT_PRESENTATION, name)).convert("RGBA")
    return CACHE[name]


def scene_background(scene: dict) -> Image.Image:
    if scene["type"] == "detect":
        return blurred_background(f"photo_{scene['photo']}", PHOTOS[scene["photo"]], PHOTO_BLUR, BACKGROUND_DARKEN)
    return blurred_background("background", presentation_image(BACKGROUND_IMAGE), BACKGROUND_BLUR, BACKGROUND_DARKEN, BACKGROUND_ANCHOR_X)


def new_frame() -> Image.Image:
    return CURRENT_BACKGROUND.copy()


def is_rtl() -> bool:
    return LANG["direction"] == "rtl"


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, spacing: float) -> float:
    width = draw.textlength(text, font=fnt, direction=LANG["direction"])
    if spacing > 0 and not is_rtl():
        width += spacing * (len(text) - 1)
    return width


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: float, spacing: float) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if text_width(draw, candidate, fnt, spacing) <= max_width or not line:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def draw_line(draw: ImageDraw.ImageDraw, x: float, y: float, line: str, fnt: ImageFont.FreeTypeFont, fill: tuple, spacing: float) -> None:
    if spacing <= 0 or is_rtl():
        draw.text((x, y), line, font=fnt, fill=fill, direction=LANG["direction"])
        return
    for char in line:
        draw.text((x, y), char, font=fnt, fill=fill)
        x += draw.textlength(char, font=fnt) + spacing


def draw_blocks(img: Image.Image, blocks: list[dict], center_y: float | None = None, bottom: float | None = None, box: bool = True, alpha: float = 1.0, top: float | None = None) -> None:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad = unit() * 0.025
    max_text_w = (box_max_width() if box else WIDTH * 0.85) - 2 * pad
    laid_out = []
    for block in blocks:
        spacing = block.get("spacing", 0.0) * block["font"].size
        lines = wrap(draw, block["text"], block["font"], max_text_w, spacing)
        laid_out.append((block, lines, spacing))
    widths = [text_width(draw, line, b["font"], s) for b, lines, s in laid_out for line in lines]
    heights = [b["font"].size * 1.35 * len(lines) for b, lines, _ in laid_out]
    box_w = max(widths) + 2 * pad
    box_h = sum(heights) + 2 * pad - blocks[-1]["font"].size * 0.2
    if center_y is None and top is not None:
        center_y = top + box_h / 2
    elif center_y is None:
        center_y = bottom - box_h / 2
    x0 = (WIDTH - box_w) / 2
    y0 = center_y - box_h / 2
    if box:
        draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=round(unit() * 0.02), fill=(0, 0, 0, round(255 * BOX_ALPHA * alpha)))
    y = y0 + pad
    for block, lines, spacing in laid_out:
        a = alpha * block.get("alpha", 1.0)
        for line in lines:
            w = text_width(draw, line, block["font"], spacing)
            x = (WIDTH - w) / 2
            if not box:
                draw_line(draw, x + 2, y + 2, line, block["font"], (0, 0, 0, round(150 * a)), spacing)
            draw_line(draw, x, y, line, block["font"], block["color"] + (round(255 * a),), spacing)
            y += block["font"].size * 1.35
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


def title_text(text: str) -> str:
    return text if is_rtl() else text.upper()


def header_center() -> float:
    return safe_top() + unit() * 0.11 - HEIGHT * PORTRAIT_HEADER_SHIFT


def header_bottom() -> float:
    title_font = font(0.044)
    box_h = title_font.size * 1.35 + 2 * unit() * 0.025 - title_font.size * 0.2
    return header_center() + box_h / 2


def between_captions() -> float:
    return (header_bottom() + caption_top()) / 2


def content_top() -> float:
    return header_center() + unit() * 0.08


def draw_title(img: Image.Image, text: str, alpha: float = 1.0) -> None:
    block = {"text": title_text(text), "font": font(0.044), "color": TEXT_COLOR, "spacing": TITLE_SPACING}
    center_y = header_center() if is_portrait() else HEIGHT * 0.075
    draw_blocks(img, [block], center_y=center_y, alpha=alpha)


def caption_top() -> float:
    return safe_bottom() - unit() * 0.23 + HEIGHT * PORTRAIT_CAPTION_SHIFT


def content_bottom() -> float:
    return safe_bottom() - unit() * 0.25 + HEIGHT * PORTRAIT_CONTENT_SHIFT


def draw_caption(img: Image.Image, text: str, alpha: float = 1.0, note: str | None = None, note_alpha: float = 1.0, top: float | None = None) -> None:
    blocks = [{"text": text, "font": font(0.042), "color": TEXT_COLOR}]
    if note is not None:
        blocks.append({"text": note, "font": font(0.034, bold=False), "color": TEXT_COLOR, "alpha": note_alpha})
    if is_portrait() and top is None:
        top = caption_top()
    draw_blocks(img, blocks, bottom=HEIGHT * 0.97, alpha=alpha, top=top)


def draw_text_block(img: Image.Image, text: str, fnt: ImageFont.FreeTypeFont, center_y: float, color: tuple, alpha: float = 1.0) -> None:
    draw_blocks(img, [{"text": text, "font": fnt, "color": TEXT_COLOR}], center_y=center_y, box=False, alpha=alpha)


def corner_logo_image() -> Image.Image:
    logo, _ = fit(presentation_image(CORNER_LOGO), WIDTH, unit() * CORNER_LOGO_HEIGHT)
    return logo


def box_max_width() -> float:
    if is_portrait():
        return WIDTH * 0.88
    margin = unit() * 0.03
    reserved = margin + corner_logo_image().width + margin
    return min(WIDTH * BOX_MAX_WIDTH, WIDTH - 2 * reserved)


def draw_corner_logo(img: Image.Image) -> None:
    logo = corner_logo_image()
    margin = unit() * 0.03
    img.paste(logo, (round(margin), round(HEIGHT - margin - logo.height)), logo)


def draw_year_tag(draw: ImageDraw.ImageDraw, year: str, x: float, y: float) -> None:
    fnt = font(0.04)
    pad = unit() * 0.012
    w = draw.textlength(year, font=fnt)
    draw.rounded_rectangle([x, y, x + w + 2 * pad, y + fnt.size * 1.25 + pad], radius=round(unit() * 0.01), fill=(0, 0, 0, round(255 * BOX_ALPHA)))
    draw.text((x + pad, y + pad * 0.4), year, font=fnt, fill=TEXT_COLOR)


def place_photo(img: Image.Image, zoom: float) -> tuple[Image.Image, float, float, float]:
    scale = min(WIDTH / img.width, HEIGHT / img.height) * zoom
    w = round(img.width * scale)
    h = round(img.height * scale)
    x = (WIDTH - w) / 2
    center_y = content_bottom() - h / 2 if is_portrait() else HEIGHT / 2
    y = center_y - h / 2
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
def scene_intro(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    a0 = progress(t, 0.1, 0.9)
    a1 = progress(t, 1.0, 1.8)
    a2 = progress(t, 1.7, 2.5)
    if is_portrait():
        positions = [HEIGHT * 0.27, HEIGHT * 0.43, HEIGHT * 0.53]
    else:
        positions = [HEIGHT * 0.22, HEIGHT * 0.48, HEIGHT * 0.66]
    logo, _ = fit(LOGO_IMAGE, WIDTH * 0.7 if is_portrait() else WIDTH * 0.4, unit() * 0.22)
    logo.putalpha(logo.getchannel("A").point(lambda v: round(v * a0)))
    img.paste(logo, (round((WIDTH - logo.width) / 2), round(positions[0] - logo.height / 2)), logo)
    draw_text_block(img, text(scene, "title"), font(0.1), positions[1], ACCENT, a1)
    draw_text_block(img, text(scene, "text"), font(0.05, bold=False), positions[2], TEXT_COLOR, a2)
    return img


def scene_outro(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    a1 = progress(t, 0.2, 1.0)
    a2 = progress(t, 0.9, 1.7)
    a3 = progress(t, 1.6, 2.4)
    url_font = ImageFont.truetype(LANGUAGES["en"]["font_bold"], round(unit() * (0.06 if is_portrait() else 0.035)))
    if is_portrait():
        draw_text_block(img, text(scene, "title"), font(0.075), HEIGHT * 0.28, ACCENT, a1)
        draw_text_block(img, text(scene, "text"), font(0.05, bold=False), HEIGHT * 0.41, TEXT_COLOR, a2)
        draw_text_block(img, SUBMISSION_LABEL, url_font, HEIGHT * 0.57, ACCENT, a3)
        return img
    draw_text_block(img, text(scene, "title"), font(0.08), HEIGHT * 0.12, ACCENT, a1)
    draw_text_block(img, text(scene, "text"), font(0.045, bold=False), HEIGHT * 0.3, TEXT_COLOR, a2)
    size = round(HEIGHT * 0.38)
    qr = QR_IMAGE.resize((size, size), Image.NEAREST)
    qr = Image.blend(new_frame().crop((0, 0, size, size)), qr, a3)
    img.paste(qr, (round((WIDTH - size) / 2), round(HEIGHT * 0.45)))
    label_y = HEIGHT * 0.45 + size + HEIGHT * 0.05
    draw_text_block(img, text(scene, "qr"), font(0.035), label_y, TEXT_COLOR, a3)
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


def scene_detect(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    boxes_at = scene.get("boxes_at", 0.0)
    zoom = 1.0 + 0.05 * ease(t / boxes_at) if boxes_at > 0 else 1.05
    photo, scale, ox, oy = place_photo(PHOTOS[scene["photo"]], zoom)
    img.paste(photo, (round(ox), round(oy)))

    detections = sorted(ASSETS["detections"][scene["photo"]], key=lambda d: list(LABEL_TO_PART).index(d["label"]))
    if "parts" in scene:
        detections = [d for d in detections if LABEL_TO_PART[d["label"]] in scene["parts"]]
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fnt = font(0.035)
    step = min(0.6, (scene["duration"] - boxes_at - 1.5) / max(1, len(detections)))
    for i, det in enumerate(detections):
        p = progress(t, boxes_at + 0.3 + i * step, boxes_at + 0.3 + i * step + 0.4)
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
        draw.rectangle([cx - hw, cy - hh, cx + hw, cy + hh], outline=color + (round(255 * p),), width=max(3, round(unit() / 270)))
        name = part_name(det["label"])
        tw = draw.textlength(name, font=fnt, direction=LANG["direction"])
        tag_w = tw + fnt.size * 0.6
        tx = min(max(cx - hw, 0), WIDTH - tag_w)
        ty = cy - hh - fnt.size * 1.4
        draw.rectangle([tx, ty, tx + tag_w, ty + fnt.size * 1.4], fill=color + (round(230 * p),))
        draw.text((tx + fnt.size * 0.3, ty + fnt.size * 0.1), name, font=fnt, fill=(0, 0, 0, round(255 * p)), direction=LANG["direction"])
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    if is_portrait() and "title" in LANG["texts"][scene["id"]]:
        draw_title(img, text(scene, "title"), progress(t, 0.1, 0.8))
    if "photo_id" in scene and t < boxes_at:
        photo_scene = {"id": scene["photo_id"]}
        draw_caption(img, text(photo_scene, "text"), progress(t, 0.3, 1.0) * (1 - progress(t, boxes_at - 0.3, boxes_at)))
    else:
        draw_caption(img, text(scene, "text"), progress(t, boxes_at, boxes_at + 0.3) if boxes_at > 0 else 1.0)
    return img


def crop_to_keypoints(img: Image.Image, kpts: np.ndarray, margin: float, crop_x: bool = True, crop_y: bool = True) -> tuple[Image.Image, np.ndarray]:
    x0, y0 = kpts.min(axis=0)
    x1, y1 = kpts.max(axis=0)
    pad_x = (x1 - x0) * margin + img.width * 0.03
    pad_y = (y1 - y0) * margin + img.height * 0.03
    box = (
        max(0, round(x0 - pad_x)) if crop_x else 0,
        max(0, round(y0 - pad_y)) if crop_y else 0,
        min(img.width, round(x1 + pad_x)) if crop_x else img.width,
        min(img.height, round(y1 + pad_y)) if crop_y else img.height,
    )
    return img.crop(box), kpts - np.array([box[0], box[1]])


def crop_part(img: Image.Image, kpts: np.ndarray, part: str) -> tuple[Image.Image, np.ndarray]:
    mode = PORTRAIT_CROP.get(part, "")
    if not mode:
        return img, kpts
    return crop_to_keypoints(img, kpts, PORTRAIT_CROP_MARGIN, crop_x="x" in mode, crop_y="y" in mode)


def layout_pair_row(orig1: Image.Image, orig2: Image.Image, top: float, bottom: float) -> tuple[Image.Image, Image.Image, float, float, float, float]:
    gap = WIDTH * 0.04
    max_w = WIDTH * 0.9 - gap
    max_h = bottom - top
    aspect1 = orig1.width / orig1.height
    aspect2 = orig2.width / orig2.height
    crop_h = min(max_h, max_w / (aspect1 + aspect2))
    crop1 = orig1.resize((round(aspect1 * crop_h), round(crop_h)), Image.LANCZOS)
    crop2 = orig2.resize((round(aspect2 * crop_h), round(crop_h)), Image.LANCZOS)
    total_w = crop1.width + gap + crop2.width
    center_y = (top + bottom) / 2
    x1 = (WIDTH - total_w) / 2
    y1 = center_y - crop_h / 2
    x2 = x1 + crop1.width + gap
    y2 = y1
    return crop1, crop2, x1, y1, x2, y2


def layout_pair_column(orig1: Image.Image, orig2: Image.Image, top: float, bottom: float, gap_rel: float = 0.04, max_w_rel: float = 0.9) -> tuple[Image.Image, Image.Image, float, float, float, float]:
    gap = unit() * gap_rel
    max_w = WIDTH * max_w_rel
    max_h = bottom - top - gap
    aspect1 = orig1.width / orig1.height
    aspect2 = orig2.width / orig2.height
    crop_w = min(max_w, max_h / (1 / aspect1 + 1 / aspect2))
    crop1 = orig1.resize((round(crop_w), round(crop_w / aspect1)), Image.LANCZOS)
    crop2 = orig2.resize((round(crop_w), round(crop_w / aspect2)), Image.LANCZOS)
    total_h = crop1.height + gap + crop2.height
    x1 = (WIDTH - crop_w) / 2
    y1 = (top + bottom - total_h) / 2
    x2 = x1
    y2 = y1 + crop1.height + gap
    return crop1, crop2, x1, y1, x2, y2


def scene_match(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    match = MATCHES[scene["part"]]
    orig1 = CROPS[match["crop1"]]
    orig2 = CROPS[match["crop2"]]
    kpts1 = np.asarray(match["kpts1"])
    kpts2 = np.asarray(match["kpts2"])
    if is_portrait():
        orig1, kpts1 = crop_part(orig1, kpts1, scene["part"])
        orig2, kpts2 = crop_part(orig2, kpts2, scene["part"])
        top = content_top()
        bottom = content_bottom()
        column = layout_pair_column(orig1, orig2, top, bottom, gap_rel=0.02, max_w_rel=0.96)
        row = layout_pair_row(orig1, orig2, top, bottom)
        column_area = column[0].width * column[0].height + column[1].width * column[1].height
        row_area = row[0].width * row[0].height + row[1].width * row[1].height
        layout = scene.get("layout", "column" if column_area >= row_area else "row")
        crop1, crop2, x1, y1, x2, y2 = column if layout == "column" else row
        content_h = max(y1 + crop1.height, y2 + crop2.height) - min(y1, y2)
        dy = between_captions() - (min(y1, y2) + content_h / 2)
        y1 += dy
        y2 += dy
    else:
        crop1, crop2, x1, y1, x2, y2 = layout_pair_row(orig1, orig2, HEIGHT * 0.19, HEIGHT * 0.81)
    s1 = crop1.height / orig1.height
    s2 = crop2.height / orig2.height
    img.paste(crop1, (round(x1), round(y1)))
    img.paste(crop2, (round(x2), round(y2)))

    draw = ImageDraw.Draw(img, "RGBA")
    if not is_portrait():
        year_fnt = font(0.04)
        for year, x, w, y in [(YEARS[0], x1, crop1.width, y1), (YEARS[1], x2, crop2.width, y2)]:
            tw = draw.textlength(year, font=year_fnt)
            draw.text((x + (w - tw) / 2, y - year_fnt.size * 1.5), year, font=year_fnt, fill=TEXT_COLOR)

    n = len(kpts1)
    order = np.argsort(kpts1[:, 1])
    colors = match_colors(n)
    start = 0.8
    end = scene["duration"] - 3.0
    r = max(2, round(unit() * 0.005))
    line_width = max(2, round(unit() / 360))
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
        draw.line([(ax, ay), (ex, ey)], fill=colors[rank], width=line_width)
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], outline=colors[rank], width=line_width)
        if p >= 1:
            draw.ellipse([bx - r, by - r, bx + r, by + r], outline=colors[rank], width=line_width)
    if is_portrait():
        margin = unit() * 0.015
        draw_year_tag(draw, YEARS[0], x1 + margin, y1 + margin)
        draw_year_tag(draw, YEARS[1], x2 + margin, y2 + margin)

    draw_title(img, text(scene, "title"), 1.0)
    note_alpha = progress(t, 1.0, 1.8)
    top = caption_top() if is_portrait() else None
    draw_caption(img, text(scene, "text").format(n=n_drawn), 1.0, note=text(scene, "note"), note_alpha=note_alpha, top=top)
    return img


def scene_summary(scene: dict, t: float) -> Image.Image:
    if is_portrait():
        return scene_summary_portrait(scene, t)
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


def scene_summary_portrait(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    draw = ImageDraw.Draw(img)
    parts = [part for part in MATCHES if part in scene.get("parts", list(MATCHES))]
    top = content_top()
    bottom = content_bottom()
    row_h = (bottom - top) / len(parts)
    pair_h = row_h * 0.8
    name_fnt = font(0.045)
    check_r = unit() * 0.035
    content_h = row_h * (len(parts) - 1) + pair_h + row_h * 0.06 + name_fnt.size * 1.3
    top = between_captions() - content_h / 2
    for i, part in enumerate(parts):
        match = MATCHES[part]
        p = progress(t, 0.3 + i * 0.5, 0.8 + i * 0.5)
        if p <= 0:
            continue
        orig1, _ = crop_part(CROPS[match["crop1"]], np.asarray(match["kpts1"]), part)
        orig2, _ = crop_part(CROPS[match["crop2"]], np.asarray(match["kpts2"]), part)
        crop1, _ = fit(orig1, WIDTH * 0.47, pair_h)
        crop2, _ = fit(orig2, WIDTH * 0.47, pair_h)
        y = top + row_h * i + pair_h / 2
        gap = WIDTH * 0.02
        img.paste(Image.blend(new_frame().crop((0, 0) + crop1.size), crop1, p), (round(WIDTH / 2 - gap / 2 - crop1.width), round(y - crop1.height / 2)))
        img.paste(Image.blend(new_frame().crop((0, 0) + crop2.size), crop2, p), (round(WIDTH / 2 + gap / 2), round(y - crop2.height / 2)))
        name = part_name(match["label1"])
        tw = draw.textlength(name, font=name_fnt, direction=LANG["direction"])
        line_y = y + max(crop1.height, crop2.height) / 2 + unit() * 0.015
        total_w = check_r * 2 + unit() * 0.02 + tw
        x0 = (WIDTH - total_w) / 2
        draw.text((x0 + check_r * 2 + unit() * 0.02, line_y), name, font=name_fnt, fill=TEXT_COLOR, direction=LANG["direction"])
        q = progress(t, 0.8 + i * 0.5, 1.3 + i * 0.5)
        if q > 0:
            draw_checkmark(draw, x0 + check_r, line_y + name_fnt.size * 0.6, check_r, q)
    draw_title(img, text(scene, "title"), progress(t, 1.8, 2.5))
    draw_caption(img, text(scene, "text"), progress(t, 1.8, 2.5), top=caption_top())
    return img


def scene_hook(scene: dict, t: float) -> Image.Image:
    img = new_frame()
    top = content_top()
    bottom = HEIGHT * 0.95
    photo1, photo2, x1, y1, x2, y2 = layout_pair_column(PHOTOS[YEARS[0]], PHOTOS[YEARS[1]], top, bottom, gap_rel=0.02, max_w_rel=0.98)
    for i, (photo, x, y) in enumerate([(photo1, x1, y1), (photo2, x2, y2)]):
        p = progress(t, 0.0 + i * 0.4, 0.5 + i * 0.4) if i else 1.0
        img.paste(Image.blend(new_frame().crop((0, 0) + photo.size), photo, p), (round(x), round(y)))
    draw = ImageDraw.Draw(img, "RGBA")
    margin = unit() * 0.015
    draw_year_tag(draw, YEARS[0], x1 + margin, y1 + margin)
    if t >= 0.4:
        draw_year_tag(draw, YEARS[1], x2 + margin, y2 + margin)
    draw_title(img, text(scene, "title"), 1.0)
    return img


def scene_endcard(scene: dict, t: float) -> Image.Image:
    img = cover(presentation_image(END_CARD_IMAGE).convert("RGB"), BACKGROUND_ANCHOR_X)
    block = {"text": title_text(text(scene, "title")), "font": font(0.09), "color": TEXT_COLOR, "spacing": TITLE_SPACING}
    draw_blocks(img, [block], center_y=header_center() + unit() * 0.01, box=False, alpha=progress(t, 0.2, 0.9))
    logo, _ = fit(presentation_image(CORNER_LOGO), WIDTH * 0.6, unit() * 0.13)
    img.paste(logo, (round((WIDTH - logo.width) / 2), round(header_center() + unit() * 0.13)), logo)
    draw = ImageDraw.Draw(img)
    credit_font = ImageFont.truetype(LANGUAGES["en"]["font_regular"], round(unit() * 0.03))
    credit_w = draw.textlength(END_CARD_CREDIT, font=credit_font)
    draw.text((WIDTH - unit() * 0.05 - credit_w, HEIGHT - unit() * 0.04 - credit_font.size * 1.3), END_CARD_CREDIT, font=credit_font, fill=(255, 255, 255))
    return img


def scene_slide(scene: dict, t: float) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    slide, _ = fit(presentation_image(scene["image"]), WIDTH, HEIGHT)
    img.paste(slide, (round((WIDTH - slide.width) / 2), round((HEIGHT - slide.height) / 2)), slide)
    return img


SCENES = {
    "hook": scene_hook,
    "endcard": scene_endcard,
    "intro": scene_intro,
    "outro": scene_outro,
    "slide": scene_slide,
    "fingerprint": scene_fingerprint,
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


def make_frame(scene: dict, t: float) -> Image.Image:
    global CURRENT_BACKGROUND
    if scene["type"] == "slide":
        CURRENT_BACKGROUND = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        return apply_fade(scene_slide(scene, t), scene, t)
    CURRENT_BACKGROUND = scene_background(scene)
    frame = SCENES[scene["type"]](scene, t)
    if scene["type"] not in ["intro", "endcard"] and not is_portrait():
        draw_corner_logo(frame)
    CURRENT_BACKGROUND = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    return apply_fade(frame, scene, t)


def storyboard() -> list[dict]:
    return STORYBOARD_PORTRAIT if is_portrait() else STORYBOARD_LANDSCAPE


def render(output: str) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for scene in storyboard():
        n_frames = round(scene["duration"] * FPS)
        for i in range(n_frames):
            process.stdin.write(make_frame(scene, i / FPS).tobytes())
    process.stdin.close()
    process.wait()


def preview(output: str) -> None:
    scenes = storyboard()
    n_cols = 6 if is_portrait() else 3
    thumb_w = WIDTH // 3
    thumb_h = HEIGHT // 3
    n_rows = (len(scenes) + n_cols - 1) // n_cols
    sheet = Image.new("RGB", (thumb_w * n_cols, thumb_h * n_rows), (255, 255, 255))
    for i, scene in enumerate(scenes):
        t = scene["duration"] - FADE - 0.1
        frame = make_frame(scene, t)
        sheet.paste(frame.resize((thumb_w, thumb_h), Image.LANCZOS), ((i % n_cols) * thumb_w, (i // n_cols) * thumb_h))
    sheet.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("langs", nargs="*", default=list(LANGUAGES))
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    WIDTH = args.width
    HEIGHT = args.height
    for lang in args.langs:
        LANG = LANGUAGES[lang]
        output = OUTPUT.format(lang=lang, width=WIDTH, height=HEIGHT)
        if args.preview:
            preview(os.path.join(ROOT, output.replace(".mp4", "_preview.png")))
        else:
            render(os.path.join(ROOT, output))
