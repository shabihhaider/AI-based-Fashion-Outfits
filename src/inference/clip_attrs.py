from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

# ---------------- Attribute vocab ----------------
ATTR_SETS: Dict[str, List[str]] = {
    "pattern": [
        "solid", "striped", "plaid", "floral", "polka dot",
        "animal print", "geometric", "camouflage", "tie-dye", "graphic"
    ],
    "fabric": [
        "denim", "cotton", "knit", "wool", "silk", "satin", "leather", "linen",
        "chiffon", "velvet", "lace", "corduroy", "polyester"
    ],
    "style": [
        "casual", "formal", "business", "streetwear", "sporty", "vintage",
        "trendy", "classic", "bohemian", "minimal", "preppy", "edgy"
    ],
    "fit": ["tight", "regular", "loose", "oversized"],
}

# Multiple prompt templates improves zero-shot accuracy a lot
TEMPLATES: Dict[str, List[str]] = {
    "pattern": [
        "a photo of clothing with a {} pattern",
        "a garment that is {}",
        "a {} patterned apparel",
        "clothes featuring {}"
    ],
    "fabric": [
        "a photo of clothing made of {}",
        "a garment in {} fabric",
        "{} material clothing",
        "clothes crafted from {}"
    ],
    "style": [
        "a {} style outfit",
        "clothing in a {} style",
        "a {} fashion look",
        "an outfit that is {}"
    ],
    "fit": [
        "a {} fit garment",
        "clothing with a {} fit",
        "a {} silhouette outfit",
        "an apparel piece that is {}"
    ],
}

def _unique(seq: List[str]) -> List[str]:
    seen = set(); out=[]
    for s in seq:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

class ZeroShotAttrModel:
    def __init__(self, device: str = "auto", model_name: str = "ViT-B-16", pretrained: str = "openai"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # Precompute text features for each attribute set
        self.text_feats: Dict[str, torch.Tensor] = {}
        self.labels: Dict[str, List[str]] = {}
        with torch.no_grad():
            for attr, labels in ATTR_SETS.items():
                labels = _unique(labels)
                self.labels[attr] = labels
                all_text_embs = []
                for lab in labels:
                    prompts = [t.format(lab) for t in TEMPLATES[attr]]
                    tok = self.tokenizer(prompts).to(self.device)
                    txt = self.model.encode_text(tok)
                    txt = F.normalize(txt, dim=-1)
                    # average the template embeddings for this label
                    all_text_embs.append(txt.mean(dim=0, keepdim=True))
                feats = torch.cat(all_text_embs, dim=0)  # (L, D)
                feats = F.normalize(feats, dim=-1)
                self.text_feats[attr] = feats  # unit-norm

    def predict_batch(self, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """
        Returns dict per attribute:
          - 'scores': Tensor [B, L] cosine-sim
        """
        with torch.no_grad():
            imgs = [self.preprocess(im.convert("RGB")) for im in images]
            im_batch = torch.stack(imgs, dim=0).to(self.device)
            im_feats = self.model.encode_image(im_batch)
            im_feats = F.normalize(im_feats, dim=-1)       # [B, D]

            outputs = {}
            for attr, txt in self.text_feats.items():      # txt: [L, D]
                # cosine similarity equals dot product due to normalization
                scores = im_feats @ txt.T                  # [B, L]
                outputs[attr] = scores
            return outputs

    def decode_topk(self, scores: torch.Tensor, attr: str, k: int = 3) -> List[List[Tuple[str, float]]]:
        """
        scores: [B, L] -> list of top-k [(label, prob)] per sample
        We softmax over labels to get a pseudo-probability.
        """
        lab = self.labels[attr]
        probs = torch.softmax(scores, dim=1)               # [B, L]
        topk = torch.topk(probs, k=min(k, probs.size(1)), dim=1)
        res = []
        for i in range(probs.size(0)):
            idx = topk.indices[i].tolist()
            sc = topk.values[i].tolist()
            res.append([(lab[j], float(sc[t])) for t, j in enumerate(idx)])
        return res
