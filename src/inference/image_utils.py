from pathlib import Path
from typing import Union, Tuple
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_image_rgb(path: Union[str, Path]) -> Image.Image:
    return Image.open(path).convert("RGB")

def preprocess_pil(im: Image.Image, img_size: int) -> np.ndarray:
    im = im.resize((img_size, img_size), resample=Image.BICUBIC)
    x = np.asarray(im, dtype=np.float32) / 255.0           # HWC, 0..1
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))                         # CHW
    x = np.expand_dims(x, 0)                               # NCHW
    return x

def is_image_file(path: Union[str, Path]) -> bool:
    return str(path).lower().endswith((".jpg",".jpeg",".png",".webp",".bmp"))
