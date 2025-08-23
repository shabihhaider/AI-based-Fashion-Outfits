import json
from pathlib import Path
from typing import List, Dict, Union
import numpy as np
import onnxruntime as ort
import yaml

from .image_utils import load_image_rgb, preprocess_pil

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

class ItemClassifier:
    def __init__(self, config_path: Union[str, Path]):
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        mcfg = cfg["model"]
        self.onnx_path = Path(mcfg["onnx_path"])
        self.meta_path = Path(mcfg["meta_path"])
        self.topk = int(mcfg.get("topk", 3))
        providers = mcfg.get("providers", ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.classes: List[str] = meta["classes"]
        self.img_size: int = int(meta.get("img_size", 512))
        # input / output names (from export)
        self.input_name  = self.session.get_inputs()[0].name   # "image"
        self.output_name = self.session.get_outputs()[0].name  # "logits"

    def predict_one(self, image_path: Union[str, Path]) -> Dict:
        im = load_image_rgb(image_path)
        x = preprocess_pil(im, self.img_size)                  # NCHW float32
        logits = self.session.run([self.output_name], {self.input_name: x})[0]  # (1, C)
        probs = softmax(logits, axis=1)[0]
        topk_idx = probs.argsort()[-self.topk:][::-1]
        result = [{
            "label": self.classes[i],
            "prob": float(probs[i])
        } for i in topk_idx]
        return {
            "image": str(image_path),
            "topk": result,
            "pred_label": result[0]["label"],
            "pred_prob": result[0]["prob"]
        }
