import numpy as np
import cv2
from PIL import ImageDraw

def uncompressed_rle_to_mask(rle):
    """Decode COCO-style uncompressed RLE into a binary mask (0/1)."""
    h, w = rle["size"]
    counts = rle["counts"]

    mask = np.zeros(h * w, dtype=np.uint8)
    val = 0
    idx = 0
    for c in counts:
        mask[idx:idx + c] = val
        idx += c
        val = 1 - val
    mask = mask.reshape((h, w), order='F')
    return mask


def mask_to_yolo(mask, class_id=0):
    """Convert a binary mask (0/1) into YOLO polygon segmentation format."""
    h, w = mask.shape

    # ensure 8-bit binary mask
    mask8 = (mask * 255).astype(np.uint8)

    # find outer contours only
    contours, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_segments = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # ignore tiny noise
            continue

        contour = contour.squeeze().astype(float)
        if contour.ndim != 2:
            continue

        # normalize to [0,1]
        contour[:, 0] = contour[:, 0] / float(w)
        contour[:, 1] = contour[:, 1] / float(h)

        coords = contour.flatten().tolist()
        yolo_segments.append(f"{class_id} " + " ".join(f"{x:.6f}" for x in coords))

    return yolo_segments

def rle_to_yolo(rle, class_id=0):
    mask = uncompressed_rle_to_mask(rle)
    return mask_to_yolo(mask, class_id)

def draw_yolo_on_pil(image, yolo_segments, color=(0,255,0)):
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for seg in yolo_segments:
        parts = seg.strip().split()
        class_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        points = [(x * w, y * h) for x, y in coords]
        draw.line(points + [points[0]], fill=color, width=2)

    return img