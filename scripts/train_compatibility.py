# scripts/train_compatibility.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import timm
import yaml

# allow "src" imports when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.compatibility import ImageOnlyCompatibility, IMAGENET_MEAN, IMAGENET_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------ data ------------------------
class PairDataset(Dataset):
    def __init__(self, pairs_csv: Path, images_root: Path, img_size: int = 224, augment: bool = False):
        self.df = pd.read_csv(pairs_csv)[["img_a","img_b","label"]].dropna().reset_index(drop=True)
        self.root = Path(images_root)
        self.img_size = img_size
        self.augment = augment

    def __len__(self): return len(self.df)

    def _load(self, rel: str) -> Image.Image:
        im = Image.open(self.root / rel).convert("RGB")
        if self.augment:
            # light aug only on train
            im = im.resize((self.img_size, self.img_size), Image.BICUBIC)
            if np.random.rand() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            im = im.resize((self.img_size, self.img_size), Image.BICUBIC)
        return im

    def _to_tensor(self, im: Image.Image) -> torch.Tensor:
        arr = np.asarray(im, dtype=np.float32) / 255.0      # HWC
        arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        arr = np.transpose(arr, (2,0,1))                    # CHW
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        xa = self._to_tensor(self._load(r["img_a"]))
        xb = self._to_tensor(self._load(r["img_b"]))
        y  = float(r["label"])
        return xa, xb, torch.tensor(y, dtype=torch.float32)

class TripletDataset(Dataset):
    def __init__(self, triplets_csv: Path, images_root: Path, img_size: int = 224):
        self.df = pd.read_csv(triplets_csv)[["anchor","positive","negative"]].dropna().reset_index(drop=True)
        self.root = Path(images_root)
        self.img_size = img_size

    def __len__(self): return len(self.df)

    def _load(self, rel: str) -> Image.Image:
        im = Image.open(self.root / rel).convert("RGB")
        im = im.resize((self.img_size, self.img_size), Image.BICUBIC)
        return im

    def _to_tensor(self, im: Image.Image) -> torch.Tensor:
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        arr = np.transpose(arr, (2,0,1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        a = self._to_tensor(self._load(r["anchor"]))
        p = self._to_tensor(self._load(r["positive"]))
        n = self._to_tensor(self._load(r["negative"]))
        return a, p, n

# ------------------------ utils ------------------------
def read_paths_yaml() -> dict:
    return yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))

def set_seed(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ------------------------ training ------------------------
def train_loop(model, pair_loader, trip_loader, optimizer, scaler, device, margin=0.2, triplet_coef=0.2):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    triplet = nn.TripletMarginLoss(margin=margin, p=2)
    total = 0.0
    for i, (xa, xb, y) in enumerate(tqdm(pair_loader, desc="Train", leave=False)):
        xa, xb, y = xa.to(device), xb.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=("cuda" if device.type=="cuda" else "cpu")):
            logit = model(xa, xb)                   # [B]
            loss = bce(logit, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total += float(loss)

        # every few steps, pull a triplet batch (if provided)
        if trip_loader is not None and (i % 4 == 0):
            try:
                a, p, n = next(train_loop._trip_iter)
            except (AttributeError, StopIteration):
                train_loop._trip_iter = iter(trip_loader)
                a, p, n = next(train_loop._trip_iter)
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=("cuda" if device.type=="cuda" else "cpu")):
                fa = model.encoder(a)
                fp = model.encoder(p)
                fn = model.encoder(n)
                tloss = triplet(fa, fp, fn) * triplet_coef
            scaler.scale(tloss).backward()
            scaler.step(optimizer); scaler.update()
            total += float(tloss)

    return total / max(1, len(pair_loader))

@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for xa, xb, y in tqdm(loader, desc="Eval", leave=False):
        xa, xb = xa.to(device), xb.to(device)
        with torch.amp.autocast(device_type=("cuda" if device.type=="cuda" else "cpu")):
            logit = model(xa, xb)
        all_logits.append(logit.cpu().numpy())
        all_labels.append(y.numpy())
    y_true = np.concatenate(all_labels)
    y_logit = np.concatenate(all_logits)
    y_prob  = 1.0 / (1.0 + np.exp(-y_logit))
    y_pred  = (y_prob >= 0.5).astype(np.int32)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    return {"auc": float(auc), "acc": float(acc), "f1": float(f1)}

def export_onnx(model, img_size: int, out_path: Path):
    model.eval()
    dummy_a = torch.randn(1, 3, img_size, img_size, device=next(model.parameters()).device)
    dummy_b = torch.randn(1, 3, img_size, img_size, device=next(model.parameters()).device)
    torch.onnx.export(
        model, (dummy_a, dummy_b), str(out_path),
        input_names=["image_a", "image_b"], output_names=["logit"],
        opset_version=17, dynamic_axes={"image_a": {0: "batch"}, "image_b": {0: "batch"}, "logit": {0: "batch"}},
        do_constant_folding=True
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manif-root", default=None, help="Folder with pairs_train.csv (default: paths.yaml processed.polyvore.manifests)")
    ap.add_argument("--val-manif-root",   default=None, help="Folder with pairs_val.csv (default: same as train root)")
    ap.add_argument("--images-root",      default=None, help="Folder with images/{split}/.. (default: paths.yaml processed.polyvore.images)")

    ap.add_argument("--backbone", default="efficientnet_b0", choices=["efficientnet_b0","vit_base_patch16_224","resnet50"])
    ap.add_argument("--embed-dim", type=int, default=512)
    ap.add_argument("--img-size",  type=int, default=224)
    ap.add_argument("--batch-size",type=int, default=32)
    ap.add_argument("--epochs",    type=int, default=6)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--triplet-margin", type=float, default=0.2)
    ap.add_argument("--triplet-coef",   type=float, default=0.2)
    ap.add_argument("--use-triplet", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs/models")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = read_paths_yaml()

    # Resolve roots
    images_root = Path(args.images_root or cfg["processed"]["polyvore"]["images"])
    train_mani = Path(args.train_manif_root) if args.train_manif_root else Path(cfg["processed"]["polyvore"]["manifests"])
    val_mani   = Path(args.val_manif_root)   if args.val_manif_root   else train_mani

    # Datasets
    pairs_train = train_mani / "pairs_train.csv"
    pairs_val   = val_mani   / "pairs_val.csv"
    if not pairs_train.exists():
        raise SystemExit(f"❌ Missing {pairs_train}")
    if not pairs_val.exists():
        raise SystemExit(f"❌ Missing {pairs_val}")

    train_ds = PairDataset(pairs_train, images_root, img_size=args.img_size, augment=True)
    val_ds   = PairDataset(pairs_val,   images_root, img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    trip_loader = None
    if args.use_triplet:
        trip_train = train_mani / "triplets_train.csv"
        if trip_train.exists():
            trip_ds = TripletDataset(trip_train, images_root, img_size=args.img_size)
            trip_loader = DataLoader(trip_ds, batch_size=max(8, args.batch_size//2), shuffle=True, num_workers=2, pin_memory=True)
        else:
            print("⚠️  Triplets missing; continuing without triplet loss.")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageOnlyCompatibility(backbone=args.backbone, embed_dim=args.embed_dim).to(device)

    optim_params = [
        {"params": model.head.parameters(), "lr": args.lr},
        {"params": model.encoder.parameters(), "lr": args.lr/3.0},  # smaller LR for backbone
    ]
    optimizer = optim.AdamW(optim_params, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device_type=("cuda" if device.type=="cuda" else "cpu"))

    best_auc = -1.0
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir.parent / "logs" / "compat_train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_loop(model, train_loader, trip_loader if args.use_triplet else None, optimizer, scaler, device,
                             margin=args.triplet_margin, triplet_coef=args.triplet_coef)
        val = eval_loop(model, val_loader, device)
        print(f"train_loss={tr_loss:.4f} | val_auc={val['auc']:.4f} | val_acc={val['acc']:.4f} | val_f1={val['f1']:.4f}")

        # Save best by AUC
        if np.isfinite(val["auc"]) and val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save({
                "model": model.state_dict(),
                "backbone": args.backbone,
                "embed_dim": args.embed_dim,
                "img_size": args.img_size,
            }, out_dir / "compatibility_best.pt")
            print("✅ Saved best ->", out_dir / "compatibility_best.pt")

        # append metrics
        rec = {"epoch": epoch, "train_loss": tr_loss, **{f"val_{k}": v for k,v in val.items()}}
        try:
            old = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
        except Exception:
            old = []
        old.append(rec)
        metrics_path.write_text(json.dumps(old, indent=2), encoding="utf-8")

    # Export ONNX from best
    ck = torch.load(out_dir / "compatibility_best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    onnx_path = out_dir / "compatibility.onnx"
    export_onnx(model, ck.get("img_size", args.img_size), onnx_path)
    print("📦 Exported ONNX ->", onnx_path)

if __name__ == "__main__":
    main()
