# scripts/eval_compatibility.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# allow "src" imports when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.compatibility import ImageOnlyCompatibility, IMAGENET_MEAN, IMAGENET_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- data ----------
class PairDataset(Dataset):
    def __init__(self, pairs_csv: Path, images_root: Path, img_size: int = 224):
        self.df = pd.read_csv(pairs_csv)[["img_a","img_b","label"]].dropna().reset_index(drop=True)
        self.root = Path(images_root); self.img_size = img_size

    def __len__(self): return len(self.df)

    def _load(self, rel: str) -> Image.Image:
        im = Image.open(self.root / rel).convert("RGB")
        return im.resize((self.img_size, self.img_size), Image.BICUBIC)

    def _to_tensor(self, im: Image.Image) -> torch.Tensor:
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        arr = np.transpose(arr, (2,0,1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        xa = self._to_tensor(self._load(r["img_a"]))
        xb = self._to_tensor(self._load(r["img_b"]))
        y  = float(r["label"])
        return r["img_a"], r["img_b"], xa, xb, y

# ---------- utils ----------
def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    # avoids overflow warnings for large |x|
    # equivalent to 1 / (1 + exp(-x)) but numerically stable
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

@torch.no_grad()
def evaluate(pairs_csv: Path, images_root: Path, ckpt_path: Path, img_size: int, batch_size: int, threshold: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PairDataset(pairs_csv, images_root, img_size=img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))

    # restore model
    ck = torch.load(ckpt_path, map_location=device)
    model = ImageOnlyCompatibility(backbone=ck.get("backbone","efficientnet_b0"),
                                   embed_dim=ck.get("embed_dim",512)).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    all_logits, all_labels = [], []
    names_a, names_b = [], []
    for img_a, img_b, xa, xb, y in tqdm(dl, desc="TestEval"):
        xa, xb = xa.to(device), xb.to(device)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            logit = model(xa, xb)  # [B]
        all_logits.append(logit.detach().cpu().numpy())
        all_labels.append(y.numpy())
        names_a.extend(img_a); names_b.extend(img_b)

    y_true  = np.concatenate(all_labels)
    y_logit = np.concatenate(all_logits)
    y_prob  = sigmoid_stable(y_logit)
    y_pred  = (y_prob >= threshold).astype(np.int32)

    try: auc = roc_auc_score(y_true, y_prob)
    except Exception: auc = float("nan")
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    return {
        "metrics": {"auc": float(auc), "acc": float(acc), "f1": float(f1), "threshold": threshold},
        "scores":  pd.DataFrame({"img_a":names_a,"img_b":names_b,"label":y_true,"prob":y_prob})
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manif-root", required=True, help="folder containing pairs_test.csv")
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--ckpt", required=True, help="compatibility_best.pt")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.5, help="probability threshold for acc/F1")
    ap.add_argument("--out-dir", default="outputs/eval")
    args = ap.parse_args()

    manif = Path(args.manif_root)
    pairs_test = manif / "pairs_test.csv"
    if not pairs_test.exists():
        raise SystemExit(f"❌ Missing {pairs_test}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    res = evaluate(pairs_test, Path(args.images_root), Path(args.ckpt),
                   img_size=args.img_size, batch_size=args.batch_size, threshold=args.threshold)

    # write
    (out_dir / "compat_test_metrics.json").write_text(json.dumps(res["metrics"], indent=2), encoding="utf-8")
    res["scores"].to_csv(out_dir / "compat_test_scores.csv", index=False)
    print("✅ Wrote:", out_dir / "compat_test_metrics.json")
    print("✅ Wrote:", out_dir / "compat_test_scores.csv")
    print("Test metrics:", res["metrics"])

if __name__ == "__main__":
    main()
