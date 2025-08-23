# scripts/recommend_api.py
from __future__ import annotations
import io, json, argparse, sys, os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# project imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.compatibility import ImageOnlyCompatibility, IMAGENET_MEAN, IMAGENET_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------- helpers --------------------
NEUTRALS = {"black","white","gray","grey","beige","cream","denim","navy","tan","brown"}

def bucket_type(cat: str) -> str:
    s = "" if cat is None else str(cat)
    c = s.lower()
    if any(k in c for k in ["sneaker","boot","heel","flat","sandal","loafer","shoe"]): return "footwear"
    if any(k in c for k in ["jacket","coat","blazer","cardigan","hoodie","outer"]):    return "outerwear"
    if any(k in c for k in ["dress","gown","jumper dress"]):                            return "dress"
    if any(k in c for k in ["skirt","jeans","trouser","pant","short","legging","chino"]): return "bottoms"
    if any(k in c for k in ["top","tee","t-shirt","shirt","blouse","sweater","pullover","tank","camisole"]): return "tops"
    return "other"

def normalize_color_name(name: str) -> str:
    if not isinstance(name, str): return ""
    n = name.strip().lower()
    n = {"grey":"gray", "navy blue":"navy"}.get(n, n)
    return n

def hex_to_rgb(hexstr: str):
    if not isinstance(hexstr, str) or not hexstr: return None
    s = hexstr.strip().lstrip("#")
    if len(s) == 3: s = "".join([ch*2 for ch in s])
    if len(s) != 6: return None
    try:
        return (int(s[0:2],16), int(s[2:4],16), int(s[4:6],16))
    except Exception:
        return None

def rgb_to_hsv(r,g,b):
    r_, g_, b_ = r/255.0, g/255.0, b/255.0
    mx, mn = max(r_,g_,b_), min(r_,g_,b_)
    df = mx - mn
    if df == 0: h = 0
    elif mx == r_: h = (60 * ((g_-b_) / df) + 360) % 360
    elif mx == g_: h = (60 * ((b_-r_) / df) + 120) % 360
    else:          h = (60 * ((r_-g_) / df) + 240) % 360
    s = 0 if mx == 0 else df / mx
    v = mx
    return h, s, v

def hsv_to_family(h, s, v):
    if v < 0.18: return "black"
    if s < 0.10:
        if v > 0.85: return "white"
        return "gray"
    if h < 15 or h >= 345: return "red"
    if h < 45:  return "orange"
    if h < 70:  return "yellow"
    if h < 170: return "green"
    if h < 200: return "teal"
    if h < 225: return "cyan"
    if h < 255: return "blue"
    if h < 275: return "navy"
    if h < 300: return "purple"
    if h < 345: return "pink"
    return "other"

def hex_to_family(hexstr: str) -> str:
    rgb = hex_to_rgb(hexstr)
    if rgb is None: return ""
    h,s,v = rgb_to_hsv(*rgb)
    return hsv_to_family(h,s,v)

COMPLEMENT = {
    "red": {"green","olive","teal"},
    "green": {"magenta","purple","red"},
    "blue": {"orange","tan","brown"},
    "orange": {"blue","navy"},
    "yellow": {"purple","violet"},
    "purple": {"yellow","beige"},
    "pink": {"olive","army","green"},
    "teal": {"maroon","red"},
}
def color_compat_score(anchor_color: str, cand_color: str) -> float:
    a = normalize_color_name(anchor_color)
    b = normalize_color_name(cand_color)
    if not a or not b: return 0.0
    if a == b: return 0.5
    if a in NEUTRALS or b in NEUTRALS: return 0.3
    if a in COMPLEMENT and b in COMPLEMENT[a]: return 0.4
    return 0.0

def dominant_color_hsv_fast(image: Image.Image) -> str:
    im = image.convert("RGB").resize((48,48), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.uint8)
    hsv = np.zeros((arr.shape[0]*arr.shape[1], 3), dtype=np.float32)
    k = 0
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            r,g,b = arr[y,x]
            h,s,v = rgb_to_hsv(int(r),int(g),int(b))
            hsv[k] = (h,s,v); k += 1
    mask = hsv[:,1] >= 0.1
    if mask.sum() == 0:
        vmean = hsv[:,2].mean()
        return "white" if vmean>0.7 else ("black" if vmean<0.3 else "gray")
    h_med = float(np.median(hsv[mask,0]))
    return hsv_to_family(h_med, float(np.median(hsv[mask,1])), float(np.median(hsv[mask,2])))

def dominant_color_kmeans(image: Image.Image, k:int=3, iters:int=8) -> str:
    im = image.convert("RGB").resize((64,64), Image.BICUBIC)
    X = np.asarray(im, dtype=np.float32).reshape(-1,3)
    # init
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=min(k, X.shape[0]), replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        d2 = ((X[:,None,:]-C[None,:,:])**2).sum(axis=2)
        lab = d2.argmin(axis=1)
        for j in range(C.shape[0]):
            pts = X[lab==j]
            if len(pts): C[j] = pts.mean(axis=0)
    counts = np.bincount(lab, minlength=C.shape[0])
    j = int(counts.argmax())
    r,g,b = C[j]
    h,s,v = rgb_to_hsv(int(r),int(g),int(b))
    return hsv_to_family(h,s,v)

def load_image_preproc(img: Image.Image, img_size: int) -> torch.Tensor:
    im = img.convert("RGB").resize((img_size, img_size), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    arr = np.transpose(arr, (2,0,1))
    return torch.from_numpy(arr)

def load_image_from_path_or_bytes(src: str|bytes|Path) -> Image.Image:
    if isinstance(src, (str, Path)):
        return Image.open(src).convert("RGB")
    elif isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src)).convert("RGB")
    else:
        raise ValueError("user_image must be a file path or raw bytes")

# -------------------- core API --------------------
@torch.no_grad()
def recommend_topk(
    ckpt_path: str|Path,
    catalog_csv: str|Path,
    images_root: str|Path,
    user_image: str|bytes|Path,
    *,
    items_aug_csv: Optional[str|Path] = None,   # expects: primary_name/hex/frac (and secondary_*)
    allow_types: str = "tops,bottoms,outerwear,footwear",
    per_bucket: int = 5,
    img_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 0,
    color_weight: float = 0.1,
    anchor_color_mode: str = "auto",            # auto|hsv|km
) -> pd.DataFrame:
    """Return a DataFrame with columns: item_id, image_path, category, title, score, bucket"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    model = ImageOnlyCompatibility(backbone=ck.get("backbone","efficientnet_b0"),
                                   embed_dim=ck.get("embed_dim",512)).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    img_root = Path(images_root)

    # load catalog
    items = pd.read_csv(
        catalog_csv,
        dtype={"item_id": str, "image_path": str, "title": str, "category": str},
        keep_default_na=False, na_values=[]
    )
    for c in ["item_id","image_path","title","category"]:
        if c not in items.columns: items[c] = ""
    items = items.fillna("")
    items["bucket"] = items["category"].apply(bucket_type)

    # filter buckets
    allow = set(s.strip().lower() for s in allow_types.split(",") if s.strip())
    if allow:
        items = items[items["bucket"].isin(allow)].reset_index(drop=True)
        if len(items) == 0:
            raise RuntimeError("After filtering, catalog is empty. Relax allow_types.")

    # color enrichment from aug file
    color_lookup: Dict[str, Tuple[str,float]] = {}
    if items_aug_csv and Path(items_aug_csv).exists():
        aug = pd.read_csv(items_aug_csv, dtype=str, keep_default_na=False)
        for c in ["primary_name","primary_hex","primary_frac",
                  "secondary_name","secondary_hex","secondary_frac"]:
            if c not in aug.columns: aug[c] = ""
        def row_color_and_frac(row):
            name = normalize_color_name(row.get("primary_name",""))
            frac = row.get("primary_frac","")
            try: frac = float(frac)
            except Exception: frac = 1.0
            if not name and row.get("primary_hex",""):
                name = hex_to_family(row["primary_hex"])
            if not name and row.get("secondary_name",""):
                name = normalize_color_name(row["secondary_name"])
            if not name and row.get("secondary_hex",""):
                name = hex_to_family(row["secondary_hex"])
            if not (0.0 <= frac <= 1.0): frac = 1.0
            return name, float(frac)
        tmp = {}
        for _, r in aug.iterrows():
            tmp[str(r.get("image_path",""))] = row_color_and_frac(r)
        color_lookup = {ip: tmp.get(ip, ("",1.0)) for ip in items["image_path"]}

    # anchor color (auto/hsv/km)
    anc_img = load_image_from_path_or_bytes(user_image)
    if anchor_color_mode == "km":
        anchor_color = dominant_color_kmeans(anc_img, k=3, iters=8)
    elif anchor_color_mode == "hsv":
        anchor_color = dominant_color_hsv_fast(anc_img)
    else:
        anchor_color = dominant_color_hsv_fast(anc_img)  # fast & stable default

    # encode user image
    anc_tensor = load_image_preproc(anc_img, img_size).unsqueeze(0).to(device)
    with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
        fa = model.encoder(anc_tensor)

    # dataset for catalog
    class CatalogDS(Dataset):
        def __init__(self, df, root, size):
            self.df = df.reset_index(drop=True); self.root=Path(root); self.size=size
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r = self.df.iloc[i]
            x = Image.open(self.root / r["image_path"]).convert("RGB")
            x = load_image_preproc(x, self.size)
            return str(r["item_id"]), str(r["image_path"]), str(r["category"]), str(r["title"]), x

    ds = CatalogDS(items, img_root, img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=max(0, num_workers), pin_memory=(device.type=="cuda"))

    rows = []
    for ids, rels, cats, titles, xb in tqdm(dl, desc="Scoring"):
        xb = xb.to(device)
        with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
            fb = model.encoder(xb)
            B = fb.shape[0]
            logits = model.head(fa.repeat(B,1), fb)
            probs  = torch.sigmoid(logits).detach().cpu().numpy()

        # color bonus (dominance-weighted if available)
        if color_lookup and color_weight > 0:
            for i in range(B):
                cname, frac = color_lookup.get(str(rels[i]), ("",1.0))
                bonus = color_weight * color_compat_score(anchor_color, cname) * max(0.5, float(frac or 0.0))
                probs[i] = float(probs[i]) + bonus

        for i in range(B):
            rows.append((str(ids[i]), str(rels[i]), str(cats[i]), str(titles[i]), float(probs[i])))

    out = pd.DataFrame(rows, columns=["item_id","image_path","category","title","score"])
    out["bucket"] = out["category"].apply(bucket_type)

    # diversify by bucket and return
    pieces = []
    for b, grp in out.groupby("bucket"):
        pieces.append(grp.sort_values("score", ascending=False).head(max(1, per_bucket)))
    top = pd.concat(pieces, ignore_index=True).sort_values("score", ascending=False).head(20).reset_index(drop=True)
    return top

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--catalog-csv", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--user-image", required=True)
    ap.add_argument("--items-aug-csv", default="")
    ap.add_argument("--allow-types", default="tops,bottoms,outerwear,footwear")
    ap.add_argument("--per-bucket", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--color-weight", type=float, default=0.1)
    ap.add_argument("--anchor-color-mode", choices=["auto","hsv","km"], default="auto")
    ap.add_argument("--out-csv", default="outputs/eval/api_demo.csv")
    args = ap.parse_args()

    df = recommend_topk(
        ckpt_path=args.ckpt,
        catalog_csv=args.catalog_csv,
        images_root=args.images_root,
        user_image=args.user_image,
        items_aug_csv=(args.items_aug_csv or None),
        allow_types=args.allow_types,
        per_bucket=args.per_bucket,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        color_weight=args.color_weight,
        anchor_color_mode=args.anchor_color_mode,
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("✅ wrote:", args.out_csv)
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
