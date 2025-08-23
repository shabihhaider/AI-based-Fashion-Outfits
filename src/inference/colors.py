from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from PIL import Image
import colorsys

# A compact named palette (HSV prototypes + HEX for UI)
# Hue values in [0,1]; tuned for fashion-ish tones
PALETTE = [
    ("black",   (0.0, 0.0, 0.05),  "#000000"),
    ("white",   (0.0, 0.0, 0.98),  "#ffffff"),
    ("gray",    (0.0, 0.0, 0.50),  "#808080"),
    ("beige",   (0.12, 0.30, 0.90), "#f5f5dc"),
    ("brown",   (0.08, 0.75, 0.45), "#8b4513"),
    ("red",     (0.00, 0.85, 0.75), "#e53935"),
    ("maroon",  (0.00, 0.85, 0.35), "#800000"),
    ("orange",  (0.07, 0.90, 0.95), "#fb8c00"),
    ("gold",    (0.12, 0.85, 0.85), "#d4af37"),
    ("yellow",  (0.15, 0.90, 0.98), "#fdd835"),
    ("olive",   (0.20, 0.75, 0.45), "#808000"),
    ("green",   (0.33, 0.75, 0.70), "#43a047"),
    ("teal",    (0.48, 0.60, 0.60), "#008080"),
    ("cyan",    (0.50, 0.60, 0.85), "#26c6da"),
    ("blue",    (0.60, 0.85, 0.80), "#1e88e5"),
    ("navy",    (0.63, 0.75, 0.35), "#000080"),
    ("purple",  (0.75, 0.70, 0.70), "#8e24aa"),
    ("magenta", (0.85, 0.80, 0.80), "#d81b60"),
    ("pink",    (0.93, 0.40, 0.98), "#ff80ab"),
    ("silver",  (0.00, 0.00, 0.80), "#c0c0c0"),
]

PALETTE_HSV = np.array([p[1] for p in PALETTE], dtype=np.float32)

def _hue_dist(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Circular distance on hue in [0,1]."""
    d = np.abs(h1 - h2)
    return np.minimum(d, 1.0 - d)

def _rgb_to_hsv_batch(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1], shape (N,3) -> hsv in [0,1], shape (N,3)."""
    # vectorized via list-comprehension; N~16k max (fast enough)
    return np.array([colorsys.rgb_to_hsv(*row) for row in rgb], dtype=np.float32)

def _foreground_mask(hsv: np.ndarray) -> np.ndarray:
    """
    Remove studio backgrounds:
      - very low saturation & very high value => white bg
      - very low value => black
    """
    s = hsv[:,1]; v = hsv[:,2]
    mask = ~(((s < 0.12) & (v > 0.85)) | (v < 0.05))
    # if we masked too much, fall back to using all pixels
    if mask.mean() < 0.05:
        return np.ones(len(hsv), dtype=bool)
    return mask

def _resize_and_sample(im: Image.Image, max_side: int = 128) -> np.ndarray:
    im = im.convert("RGB").resize((max_side, max_side), resample=Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr.reshape(-1, 3)  # (N,3)

def extract_colors_from_image(im: Image.Image) -> Dict:
    """
    Returns:
      {
        'primary_name', 'primary_hex', 'primary_frac',
        'secondary_name', 'secondary_hex', 'secondary_frac',
        'brightness' (0..1), 'saturation' (0..1)
      }
    """
    rgb = _resize_and_sample(im)                       # (N,3) in [0,1]
    hsv = _rgb_to_hsv_batch(rgb)                      # (N,3)
    mask = _foreground_mask(hsv)
    hsv_f = hsv[mask]
    if hsv_f.size == 0:
        hsv_f = hsv

    # match to palette by weighted HSV distance
    wH, wS, wV = 2.0, 1.0, 0.5
    H = hsv_f[:,0:1]; S = hsv_f[:,1:2]; V = hsv_f[:,2:3]
    d = (wH*_hue_dist(H, PALETTE_HSV[None,:,0]))**2 + ((wS*(S-PALETTE_HSV[None,:,1]))**2) + ((wV*(V-PALETTE_HSV[None,:,2]))**2)
    idx = d.argmin(axis=1)

    # histogram
    counts = np.bincount(idx, minlength=len(PALETTE))
    freqs = counts / counts.sum() if counts.sum() > 0 else counts
    order = np.argsort(freqs)[::-1]

    # top-1
    p1 = order[0]
    p2 = order[1] if len(order) > 1 else None
    primary_name, primary_hex = PALETTE[p1][0], PALETTE[p1][2]
    primary_frac = float(freqs[p1])

    # enforce a minimum for secondary
    secondary_name = secondary_hex = ""
    secondary_frac = 0.0
    if p2 is not None and freqs[p2] >= 0.05:
        secondary_name, secondary_hex = PALETTE[p2][0], PALETTE[p2][2]
        secondary_frac = float(freqs[p2])

    # simple global stats for later features
    brightness = float(hsv_f[:,2].mean())
    saturation = float(hsv_f[:,1].mean())

    return {
        "primary_name": primary_name,
        "primary_hex": primary_hex,
        "primary_frac": round(primary_frac, 4),
        "secondary_name": secondary_name,
        "secondary_hex": secondary_hex,
        "secondary_frac": round(secondary_frac, 4),
        "brightness": round(brightness, 4),
        "saturation": round(saturation, 4),
    }
