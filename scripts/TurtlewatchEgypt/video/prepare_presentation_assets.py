import argparse
import os
import subprocess
import tempfile
import zipfile

from PIL import Image
from pptx import Presentation

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_ASSETS = os.path.join(ROOT, "assets", "presentation")
SLIDE_WIDTH = 1920
SLIDE_HEIGHT = 1080
HIDE_ON_LAST_SLIDE = ["Group 5"]
MEDIA = {"image25.png": "background.png", "image13.png": "logo_white.png"}


def extract_media(file_name: str) -> None:
    with zipfile.ZipFile(file_name) as archive:
        for source, target in MEDIA.items():
            with open(os.path.join(ROOT_ASSETS, target), "wb") as file:
                file.write(archive.read(f"ppt/media/{source}"))


def hide_shapes(file_name: str, output: str) -> None:
    presentation = Presentation(file_name)
    slide = presentation.slides[-1]
    for shape in list(slide.shapes):
        if shape.name in HIDE_ON_LAST_SLIDE:
            element = shape.element
            element.getparent().remove(element)
    presentation.save(output)


def render_slides(file_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pptx = os.path.join(tmp, "deck.pptx")
        hide_shapes(file_name, pptx)
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, pptx], check=True, capture_output=True)
        pdf = os.path.join(tmp, "deck.pdf")
        subprocess.run(["pdftoppm", "-png", "-scale-to-x", str(SLIDE_WIDTH), "-scale-to-y", str(SLIDE_HEIGHT), pdf, os.path.join(tmp, "slide")], check=True)
        slides = sorted(f for f in os.listdir(tmp) if f.startswith("slide") and f.endswith(".png"))
        for key, name in [("first", slides[0]), ("last", slides[-1])]:
            Image.open(os.path.join(tmp, name)).convert("RGB").save(os.path.join(ROOT_ASSETS, f"slide_{key}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pptx")
    args = parser.parse_args()

    os.makedirs(ROOT_ASSETS, exist_ok=True)
    extract_media(args.pptx)
    render_slides(args.pptx)
