
from __future__ import annotations
import argparse, sys, os, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.compatibility import ImageOnlyCompatibility, IMAGENET_MEAN, IMAGENET_STD
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

# ----- optional k-means for anchor color (fast, tiny) -----
def dominant_color_kmeans(image: Image.Image, k:int=3, iters:int=8) -> str:
    im = image.convert("RGB").resize((64,64), Image.BICUBIC)
    X = np.asarray(im, dtype=np.float32).reshape(-1,3)
    # init: random samples
    idx = np.random.choice(X.shape[0], size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        # assign
        d2 = ((X[:,None,:]-C[None,:,:])**2).sum(axis=2)
        lab = d2.argmin(axis=1)
        # update
        for j in range(k):
            pts = X[lab==j]
            if len(pts): C[j] = pts.mean(axis=0)
    # pick largest cluster
    counts = np.bincount(lab, minlength=k)
    j = int(counts.argmax())
    r,g,b = C[j]
    h,s,v = rgb_to_hsv(int(r),int(g),int(b))
    return hsv_to_family(h,s,v)

def load_image_tensor(path: Path, img_size: int):
    im = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    arr = np.transpose(arr, (2,0,1))
    return torch.from_numpy(arr), im  # return PIL too for k-means

class CatalogDS(Dataset):
    def __init__(self, items_df: pd.DataFrame, images_root: Path, img_size: int):
        self.df = items_df.reset_index(drop=True)
        self.root = Path(images_root); self.img_size = img_size
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        item_id = "" if pd.isna(r.get("item_id")) else str(r.get("item_id"))
        rel     = "" if pd.isna(r.get("image_path")) else str(r.get("image_path"))
        cat     = "" if pd.isna(r.get("category")) else str(r.get("category"))
        title   = "" if pd.isna(r.get("title")) else str(r.get("title"))
        x, _ = load_image_tensor(self.root / rel, self.img_size)
        return item_id, rel, cat, title, x

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--items-csv", required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--allow-types", default="", help="comma list: tops,bottoms,dress,outerwear,footwear")
    ap.add_argument("--per-bucket", type=int, default=5)
    # color options
    ap.add_argument("--items-aug-csv", default="", help="items_*_aug.csv (primary_name/hex/frac supported)")
    ap.add_argument("--anchor-in-manif-csv", default="", help="items_* csv for anchor lookup (optional)")
    ap.add_argument("--anchor-color-mode", choices=["auto","hsv","km"], default="auto",
                    help="auto: use aug if present else HSV median; hsv: force HSV; km: k-means")
    ap.add_argument("--color-weight", type=float, default=0.1)
    ap.add_argument("--out-csv", default="outputs/eval/demo_recs.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    model = ImageOnlyCompatibility(backbone=ck.get("backbone","efficientnet_b0"),
                                   embed_dim=ck.get("embed_dim",512)).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    img_root = Path(args.images_root)
    anc_path = Path(args.anchor)
    if not anc_path.exists():
        anc_path = img_root / args.anchor
    if not anc_path.exists():
        raise SystemExit(f"❌ Anchor image not found: {args.anchor}")

    items = pd.read_csv(
        args.items_csv,
        dtype={"item_id": str, "image_path": str, "title": str, "category": str},
        keep_default_na=False, na_values=[]
    )
    for c in ["item_id","image_path","title","category"]:
        if c not in items.columns: items[c] = ""
    items[["item_id","image_path","title","category"]] = items[["item_id","image_path","title","category"]].fillna("")
    items["bucket"] = items["category"].apply(bucket_type)

    allow = set(s.strip().lower() for s in args.allow_types.split(",") if s.strip())
    if allow:
        items = items[items["bucket"].isin(allow)].reset_index(drop=True)
        if len(items) == 0:
            raise SystemExit("❌ After filtering, catalog is empty. Relax --allow-types.")

    # ---- color enrichment from your aug (primary_* preferred) ----
    anchor_color = ""
    color_lookup = {}   # image_path -> (name:str, frac:float)
    if args.items_aug_csv and os.path.exists(args.items_aug_csv):
        aug = pd.read_csv(args.items_aug_csv, dtype=str, keep_default_na=False)
        # normalize expected cols
        for c in ["primary_name","primary_hex","primary_frac",
                  "secondary_name","secondary_hex","secondary_frac"]:
            if c not in aug.columns: aug[c] = ""
        # merge top color into items
        def row_color_and_frac(row):
            name = normalize_color_name(row.get("primary_name",""))
            frac = row.get("primary_frac","")
            try: frac = float(frac)
            except Exception: frac = 1.0
            if not name:
                hx = row.get("primary_hex","")
                name = hex_to_family(hx) if hx else ""
            # if still missing, try secondary
            if not name:
                name = normalize_color_name(row.get("secondary_name",""))
                if not name and row.get("secondary_hex",""):
                    name = hex_to_family(row["secondary_hex"])
            if not (0.0 <= frac <= 1.0): frac = 1.0
            return name, float(frac)
        # build lookup on the aug side
        a_lookup = {}
        for _,r in aug.iterrows():
            name, frac = row_color_and_frac(r)
            a_lookup[str(r.get("image_path",""))] = (name, frac)
        # attach to candidate items
        color_lookup = {ip: a_lookup.get(ip, ("",1.0)) for ip in items["image_path"]}

        # anchor color if anchor row present
        rel = str(anc_path.relative_to(img_root)).replace("\\","/") if str(anc_path).startswith(str(img_root)) else args.anchor
        if rel in a_lookup:
            anchor_color = a_lookup[rel][0]

    # Compute anchor color if needed
    _, anc_pil = load_image_tensor(anc_path, args.img_size)
    if not anchor_color or args.anchor_color_mode in {"hsv","km"}:
        if args.anchor_color_mode == "km":
            anchor_color = dominant_color_kmeans(anc_pil, k=3, iters=8)
        else:
            # HSV median fallback
            im = anc_pil.convert("RGB").resize((48,48), Image.BICUBIC)
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
                anchor_color = "white" if vmean>0.7 else ("black" if vmean<0.3 else "gray")
            else:
                h_med = float(np.median(hsv[mask,0]))
                anchor_color = hsv_to_family(h_med, float(np.median(hsv[mask,1])), float(np.median(hsv[mask,2])))

    # dataset/loader
    ds = CatalogDS(items, img_root, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=max(0, args.num_workers), pin_memory=(device.type=="cuda"))

    # encode anchor once
    anc_tensor, _ = load_image_tensor(anc_path, args.img_size)
    anc_tensor = anc_tensor.unsqueeze(0).to(device)
    with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
        fa = model.encoder(anc_tensor)

    rows = []
    for ids, rels, cats, titles, xb in tqdm(dl, desc="Scoring"):
        xb = xb.to(device)
        with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
            fb = model.encoder(xb)
            B = fb.shape[0]
            logits = model.head(fa.repeat(B,1), fb)
            probs  = torch.sigmoid(logits).detach().cpu().numpy()

        # color bonus (weighted by dominance fraction if present)
        if anchor_color and args.color_weight > 0:
            rel_list = [str(r) for r in rels]
            for i in range(B):
                cname, frac = color_lookup.get(rel_list[i], ("",1.0))
                bonus = args.color_weight * color_compat_score(anchor_color, cname) * max(0.5, float(frac or 0.0))
                probs[i] = float(probs[i]) + bonus

        for i in range(B):
            rows.append((str(ids[i]), str(rels[i]), str(cats[i]), str(titles[i]), float(probs[i])))

    out = pd.DataFrame(rows, columns=["item_id","image_path","category","title","score"])
    out["bucket"] = out["category"].apply(bucket_type)

    # diversify by bucket
    pieces = []
    for b, grp in out.groupby("bucket"):
        pieces.append(grp.sort_values("score", ascending=False).head(max(1, args.per_bucket)))
    top = pd.concat(pieces, ignore_index=True).sort_values("score", ascending=False).head(args.topk).reset_index(drop=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.out_csv, index=False)
    print("✅ Wrote top-k ->", args.out_csv)
    print("Anchor color (used):", anchor_color)
    print("Top-10 preview:")
    print(top.head(10).to_string(index=False))

if __name__ == "__main__":
    main()


# from pathlib import Path
# Path("/kaggle/working/scripts").mkdir(parents=True, exist_ok=True)
# Path("/kaggle/working/scripts/recommend_from_catalog.py").write_text(r"""
# from __future__ import annotations
# import argparse, sys, os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from PIL import Image, ImageFile
# from tqdm import tqdm

# import torch
# from torch.utils.data import DataLoader, Dataset

# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from src.models.compatibility import ImageOnlyCompatibility, IMAGENET_MEAN, IMAGENET_STD
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# NEUTRALS = {"black","white","gray","grey","beige","cream","denim","navy","tan","brown"}

# def bucket_type(cat: str) -> str:
#     s = "" if cat is None else str(cat)
#     c = s.lower()
#     if any(k in c for k in ["sneaker","boot","heel","flat","sandal","loafer","shoe"]): return "footwear"
#     if any(k in c for k in ["jacket","coat","blazer","cardigan","hoodie","outer"]):    return "outerwear"
#     if any(k in c for k in ["dress","gown","jumper dress"]):                            return "dress"
#     if any(k in c for k in ["skirt","jeans","trouser","pant","short","legging","chino"]): return "bottoms"
#     if any(k in c for k in ["top","tee","t-shirt","shirt","blouse","sweater","pullover","tank","camisole"]): return "tops"
#     return "other"

# COMPLEMENT = {
#     "red": {"green","olive","teal"},
#     "green": {"magenta","purple","red"},
#     "blue": {"orange","tan","brown"},
#     "orange": {"blue","navy"},
#     "yellow": {"purple","violet"},
#     "purple": {"yellow","beige"},
#     "pink": {"olive","army","green"},
#     "teal": {"maroon","red"},
# }
# def normalize_color_name(name: str) -> str:
#     if not isinstance(name, str): return ""
#     n = name.strip().lower()
#     n = {"grey":"gray", "navy blue":"navy"}.get(n, n)
#     return n

# def color_compat_score(anchor_color: str, cand_color: str) -> float:
#     a = normalize_color_name(anchor_color)
#     b = normalize_color_name(cand_color)
#     if not a or not b: return 0.0
#     if a == b: return 0.5
#     if a in NEUTRALS or b in NEUTRALS: return 0.3
#     if a in COMPLEMENT and b in COMPLEMENT[a]: return 0.4
#     return 0.0

# def load_image(path: Path, img_size: int):
#     im = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
#     arr = np.asarray(im, dtype=np.float32) / 255.0
#     arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
#     arr = np.transpose(arr, (2,0,1))
#     return torch.from_numpy(arr)

# class CatalogDS(Dataset):
#     def __init__(self, items_df: pd.DataFrame, images_root: Path, img_size: int):
#         self.df = items_df.reset_index(drop=True)
#         self.root = Path(images_root)
#         self.img_size = img_size
#     def __len__(self): return len(self.df)
#     def __getitem__(self, i):
#         r = self.df.iloc[i]
#         item_id = "" if pd.isna(r.get("item_id")) else str(r.get("item_id"))
#         rel     = "" if pd.isna(r.get("image_path")) else str(r.get("image_path"))
#         cat     = "" if pd.isna(r.get("category")) else str(r.get("category"))
#         title   = "" if pd.isna(r.get("title")) else str(r.get("title"))
#         x = load_image(self.root / rel, self.img_size)
#         return item_id, rel, cat, title, x

# @torch.no_grad()
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", required=True)
#     ap.add_argument("--images-root", required=True)
#     ap.add_argument("--items-csv", required=True)
#     ap.add_argument("--anchor", required=True)
#     ap.add_argument("--img-size", type=int, default=224)
#     ap.add_argument("--topk", type=int, default=20)
#     ap.add_argument("--batch-size", type=int, default=256)
#     ap.add_argument("--num-workers", type=int, default=2)
#     ap.add_argument("--allow-types", default="", help="comma list: tops,bottoms,dress,outerwear,footwear")
#     ap.add_argument("--per-bucket", type=int, default=5)
#     ap.add_argument("--items-aug-csv", default="", help="items_*_aug.csv with color columns")
#     ap.add_argument("--anchor-in-manif-csv", default="", help="items_* csv to find anchor for color (optional)")
#     ap.add_argument("--color-weight", type=float, default=0.1)
#     ap.add_argument("--out-csv", default="outputs/eval/demo_recs.csv")
#     args = ap.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     ck = torch.load(args.ckpt, map_location=device)
#     model = ImageOnlyCompatibility(backbone=ck.get("backbone","efficientnet_b0"),
#                                    embed_dim=ck.get("embed_dim",512)).to(device)
#     model.load_state_dict(ck["model"]); model.eval()

#     img_root = Path(args.images_root)

#     anc_path = Path(args.anchor)
#     if not anc_path.exists():
#         anc_path = img_root / args.anchor
#     if not anc_path.exists():
#         raise SystemExit(f"❌ Anchor image not found: {args.anchor}")

#     items = pd.read_csv(
#         args.items_csv,
#         dtype={"item_id": str, "image_path": str, "title": str, "category": str},
#         keep_default_na=False, na_values=[]
#     )
#     for c in ["item_id","image_path","title","category"]:
#         if c not in items.columns: items[c] = ""
#     items[["item_id","image_path","title","category"]] = items[["item_id","image_path","title","category"]].fillna("")
#     allow = set(s.strip().lower() for s in args.allow_types.split(",") if s.strip())
#     items["bucket"] = items["category"].apply(bucket_type)
#     if allow:
#         items = items[items["bucket"].isin(allow)].reset_index(drop=True)
#         if len(items) == 0:
#             raise SystemExit("❌ After filtering, catalog is empty. Relax --allow-types.")

#     # ---- color augmentation (robust to column names) ----
#     anchor_color = ""
#     color_lookup = {}
#     if args.items_aug_csv and os.path.exists(args.items_aug_csv):
#         aug = pd.read_csv(args.items_aug_csv, dtype=str, keep_default_na=False)
#         # find a usable color column
#         candidates = [
#             "dom_color_name","dominant_color_name","primary_color_name",
#             "color_name","dom_color","major_color","dominant_color"
#         ]
#         color_col = next((c for c in candidates if c in aug.columns), None)
#         if color_col is None:
#             # try hex columns (we won't map hex → name; just skip bonus)
#             hex_cands = ["dom_color_hex","dominant_color_hex","color_hex"]
#             hex_col = next((c for c in hex_cands if c in aug.columns), None)
#             if hex_col is None:
#                 print("⚠️ No recognized color column in aug file; skipping color bonus.")
#             else:
#                 print("ℹ️ Found only hex color; skipping color bonus.")
#         else:
#             if color_col != "dom_color_name":
#                 aug = aug.rename(columns={color_col: "dom_color_name"})
#             aug["dom_color_name"] = aug["dom_color_name"].fillna("")
#             # merge candidate color
#             items = items.merge(aug[["image_path","dom_color_name"]], on="image_path", how="left")
#             items["dom_color_name"] = items["dom_color_name"].fillna("")
#             # build fast lookup by image_path
#             color_lookup = dict(zip(items["image_path"], items["dom_color_name"]))
#             # try to get anchor color by relative path
#             if str(anc_path).startswith(str(img_root)):
#                 anchor_rel = str(Path(anc_path).relative_to(img_root)).replace("\\","/")
#             else:
#                 anchor_rel = args.anchor
#             arow = aug[aug["image_path"] == anchor_rel]
#             if len(arow) == 0 and args.anchor_in_manif_csv and os.path.exists(args.anchor_in_manif_csv):
#                 # fallback try: join against manifest and then aug
#                 aitems = pd.read_csv(args.anchor_in_manif_csv, dtype=str, keep_default_na=False)
#                 aitems = aitems.merge(aug[["image_path","dom_color_name"]], on="image_path", how="left")
#                 arow = aitems[aitems["image_path"].str.endswith(Path(anchor_rel).name)]
#             if len(arow):
#                 anchor_color = str(arow.iloc[0].get("dom_color_name","")).strip()

#     # dataset/loader
#     ds = CatalogDS(items, img_root, img_size=args.img_size)
#     dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
#                     num_workers=max(0, args.num_workers), pin_memory=(device.type=="cuda"))

#     # encode anchor once
#     anc_img = Image.open(anc_path).convert("RGB").resize((args.img_size, args.img_size), Image.BICUBIC)
#     arr = np.asarray(anc_img, dtype=np.float32) / 255.0
#     arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
#     anc = torch.from_numpy(np.transpose(arr, (2,0,1))).unsqueeze(0).to(device)
#     with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
#         fa = model.encoder(anc)

#     rows = []
#     for ids, rels, cats, titles, xb in tqdm(dl, desc="Scoring"):
#         xb = xb.to(device)
#         with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
#             fb = model.encoder(xb)
#             B = fb.shape[0]
#             logits = model.head(fa.repeat(B, 1), fb)
#             probs  = torch.sigmoid(logits).detach().cpu().numpy()

#         # apply color bonus if available
#         if anchor_color and color_lookup and args.color_weight > 0:
#             rel_list = [str(r) for r in rels]
#             for i in range(B):
#                 cand_color = color_lookup.get(rel_list[i], "")
#                 probs[i] = float(probs[i]) + args.color_weight * color_compat_score(anchor_color, cand_color)

#         for i in range(B):
#             rows.append((str(ids[i]), str(rels[i]), str(cats[i]), str(titles[i]), float(probs[i])))

#     out = pd.DataFrame(rows, columns=["item_id","image_path","category","title","score"])
#     out["bucket"] = out["category"].apply(bucket_type)

#     # per-bucket cap then global sort
#     pieces = []
#     for b, grp in out.groupby("bucket"):
#         pieces.append(grp.sort_values("score", ascending=False).head(max(1, args.per-bucket)))
#     top = pd.concat(pieces, ignore_index=True).sort_values("score", ascending=False).head(args.topk).reset_index(drop=True)

#     Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
#     top.to_csv(args.out_csv, index=False)
#     print("✅ Wrote top-k ->", args.out_csv)
#     print("Top-10 preview:")
#     print(top.head(10).to_string(index=False))

# if __name__ == "__main__":
#     main()
# """, encoding="utf-8")
# print("Patched recommend_from_catalog.py to accept varied color columns & fallback cleanly")


# from pathlib import Path
# Path("/kaggle/working/scripts").mkdir(parents=True, exist_ok=True)
# Path("/kaggle/working/scripts/recommend_from_catalog.py").write_text(r"""
# from __future__ import annotations
# import argparse, sys, os, json, math, random
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from PIL import Image, ImageFile
# from tqdm import tqdm

# import torch
# from torch.utils.data import DataLoader, Dataset

# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from src.models.compatibility import ImageOnlyCompatibility, IMAGENET_MEAN, IMAGENET_STD
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# NEUTRALS = {"black","white","gray","grey","beige","cream","denim","navy","tan","brown"}

# def bucket_type(cat: str) -> str:
#     s = "" if cat is None else str(cat)
#     c = s.lower()
#     if any(k in c for k in ["sneaker","boot","heel","flat","sandal","loafer","shoe"]): return "footwear"
#     if any(k in c for k in ["jacket","coat","blazer","cardigan","hoodie","outer"]):    return "outerwear"
#     if any(k in c for k in ["dress","gown","jumper dress"]):                            return "dress"
#     if any(k in c for k in ["skirt","jeans","trouser","pant","short","legging","chino"]): return "bottoms"
#     if any(k in c for k in ["top","tee","t-shirt","shirt","blouse","sweater","pullover","tank","camisole"]): return "tops"
#     return "other"

# def normalize_color_name(name: str) -> str:
#     if not isinstance(name, str): return ""
#     n = name.strip().lower()
#     n = {"grey":"gray", "navy blue":"navy"}.get(n, n)
#     return n

# def hex_to_rgb(hexstr: str):
#     if not isinstance(hexstr, str) or not hexstr: return None
#     s = hexstr.strip().lstrip("#")
#     if len(s) == 3: s = "".join([ch*2 for ch in s])
#     if len(s) != 6: return None
#     try:
#         return (int(s[0:2],16), int(s[2:4],16), int(s[4:6],16))
#     except Exception:
#         return None

# def rgb_to_hsv(r,g,b):
#     r_, g_, b_ = r/255.0, g/255.0, b/255.0
#     mx, mn = max(r_,g_,b_), min(r_,g_,b_)
#     df = mx - mn
#     if df == 0: h = 0
#     elif mx == r_: h = (60 * ((g_-b_) / df) + 360) % 360
#     elif mx == g_: h = (60 * ((b_-r_) / df) + 120) % 360
#     else:          h = (60 * ((r_-g_) / df) + 240) % 360
#     s = 0 if mx == 0 else df / mx
#     v = mx
#     return h, s, v

# def hsv_to_family(h, s, v):
#     if v < 0.18: return "black"
#     if s < 0.10:
#         if v > 0.85: return "white"
#         return "gray"
#     if h < 15 or h >= 345: return "red"
#     if h < 45:  return "orange"
#     if h < 70:  return "yellow"
#     if h < 170: return "green"
#     if h < 200: return "teal"
#     if h < 225: return "cyan"
#     if h < 255: return "blue"
#     if h < 275: return "navy"
#     if h < 300: return "purple"
#     if h < 345: return "pink"
#     return "other"

# def hex_to_family(hexstr: str) -> str:
#     rgb = hex_to_rgb(hexstr)
#     if rgb is None: return ""
#     h,s,v = rgb_to_hsv(*rgb)
#     return hsv_to_family(h,s,v)

# COMPLEMENT = {
#     "red": {"green","olive","teal"},
#     "green": {"magenta","purple","red"},
#     "blue": {"orange","tan","brown"},
#     "orange": {"blue","navy"},
#     "yellow": {"purple","violet"},
#     "purple": {"yellow","beige"},
#     "pink": {"olive","army","green"},
#     "teal": {"maroon","red"},
# }
# def color_compat_score(anchor_color: str, cand_color: str) -> float:
#     a = normalize_color_name(anchor_color)
#     b = normalize_color_name(cand_color)
#     if not a or not b: return 0.0
#     if a == b: return 0.5
#     if a in NEUTRALS or b in NEUTRALS: return 0.3
#     if a in COMPLEMENT and b in COMPLEMENT[a]: return 0.4
#     return 0.0

# # ----- optional k-means for anchor color (fast, tiny) -----
# def dominant_color_kmeans(image: Image.Image, k:int=3, iters:int=8) -> str:
#     im = image.convert("RGB").resize((64,64), Image.BICUBIC)
#     X = np.asarray(im, dtype=np.float32).reshape(-1,3)
#     # init: random samples
#     idx = np.random.choice(X.shape[0], size=k, replace=False)
#     C = X[idx].copy()
#     for _ in range(iters):
#         # assign
#         d2 = ((X[:,None,:]-C[None,:,:])**2).sum(axis=2)
#         lab = d2.argmin(axis=1)
#         # update
#         for j in range(k):
#             pts = X[lab==j]
#             if len(pts): C[j] = pts.mean(axis=0)
#     # pick largest cluster
#     counts = np.bincount(lab, minlength=k)
#     j = int(counts.argmax())
#     r,g,b = C[j]
#     h,s,v = rgb_to_hsv(int(r),int(g),int(b))
#     return hsv_to_family(h,s,v)

# def load_image_tensor(path: Path, img_size: int):
#     im = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
#     arr = np.asarray(im, dtype=np.float32) / 255.0
#     arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
#     arr = np.transpose(arr, (2,0,1))
#     return torch.from_numpy(arr), im  # return PIL too for k-means

# class CatalogDS(Dataset):
#     def __init__(self, items_df: pd.DataFrame, images_root: Path, img_size: int):
#         self.df = items_df.reset_index(drop=True)
#         self.root = Path(images_root); self.img_size = img_size
#     def __len__(self): return len(self.df)
#     def __getitem__(self, i):
#         r = self.df.iloc[i]
#         item_id = "" if pd.isna(r.get("item_id")) else str(r.get("item_id"))
#         rel     = "" if pd.isna(r.get("image_path")) else str(r.get("image_path"))
#         cat     = "" if pd.isna(r.get("category")) else str(r.get("category"))
#         title   = "" if pd.isna(r.get("title")) else str(r.get("title"))
#         x, _ = load_image_tensor(self.root / rel, self.img_size)
#         return item_id, rel, cat, title, x

# @torch.no_grad()
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", required=True)
#     ap.add_argument("--images-root", required=True)
#     ap.add_argument("--items-csv", required=True)
#     ap.add_argument("--anchor", required=True)
#     ap.add_argument("--img-size", type=int, default=224)
#     ap.add_argument("--topk", type=int, default=20)
#     ap.add_argument("--batch-size", type=int, default=256)
#     ap.add_argument("--num-workers", type=int, default=2)
#     ap.add_argument("--allow-types", default="", help="comma list: tops,bottoms,dress,outerwear,footwear")
#     ap.add_argument("--per-bucket", type=int, default=5)
#     # color options
#     ap.add_argument("--items-aug-csv", default="", help="items_*_aug.csv (primary_name/hex/frac supported)")
#     ap.add_argument("--anchor-in-manif-csv", default="", help="items_* csv for anchor lookup (optional)")
#     ap.add_argument("--anchor-color-mode", choices=["auto","hsv","km"], default="auto",
#                     help="auto: use aug if present else HSV median; hsv: force HSV; km: k-means")
#     ap.add_argument("--color-weight", type=float, default=0.1)
#     ap.add_argument("--out-csv", default="outputs/eval/demo_recs.csv")
#     args = ap.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     ck = torch.load(args.ckpt, map_location=device)
#     model = ImageOnlyCompatibility(backbone=ck.get("backbone","efficientnet_b0"),
#                                    embed_dim=ck.get("embed_dim",512)).to(device)
#     model.load_state_dict(ck["model"]); model.eval()

#     img_root = Path(args.images_root)
#     anc_path = Path(args.anchor)
#     if not anc_path.exists():
#         anc_path = img_root / args.anchor
#     if not anc_path.exists():
#         raise SystemExit(f"❌ Anchor image not found: {args.anchor}")

#     items = pd.read_csv(
#         args.items_csv,
#         dtype={"item_id": str, "image_path": str, "title": str, "category": str},
#         keep_default_na=False, na_values=[]
#     )
#     for c in ["item_id","image_path","title","category"]:
#         if c not in items.columns: items[c] = ""
#     items[["item_id","image_path","title","category"]] = items[["item_id","image_path","title","category"]].fillna("")
#     items["bucket"] = items["category"].apply(bucket_type)

#     allow = set(s.strip().lower() for s in args.allow_types.split(",") if s.strip())
#     if allow:
#         items = items[items["bucket"].isin(allow)].reset_index(drop=True)
#         if len(items) == 0:
#             raise SystemExit("❌ After filtering, catalog is empty. Relax --allow-types.")

#     # ---- color enrichment from your aug (primary_* preferred) ----
#     anchor_color = ""
#     color_lookup = {}   # image_path -> (name:str, frac:float)
#     if args.items_aug_csv and os.path.exists(args.items_aug_csv):
#         aug = pd.read_csv(args.items_aug_csv, dtype=str, keep_default_na=False)
#         # normalize expected cols
#         for c in ["primary_name","primary_hex","primary_frac",
#                   "secondary_name","secondary_hex","secondary_frac"]:
#             if c not in aug.columns: aug[c] = ""
#         # merge top color into items
#         def row_color_and_frac(row):
#             name = normalize_color_name(row.get("primary_name",""))
#             frac = row.get("primary_frac","")
#             try: frac = float(frac)
#             except Exception: frac = 1.0
#             if not name:
#                 hx = row.get("primary_hex","")
#                 name = hex_to_family(hx) if hx else ""
#             # if still missing, try secondary
#             if not name:
#                 name = normalize_color_name(row.get("secondary_name",""))
#                 if not name and row.get("secondary_hex",""):
#                     name = hex_to_family(row["secondary_hex"])
#             if not (0.0 <= frac <= 1.0): frac = 1.0
#             return name, float(frac)
#         # build lookup on the aug side
#         a_lookup = {}
#         for _,r in aug.iterrows():
#             name, frac = row_color_and_frac(r)
#             a_lookup[str(r.get("image_path",""))] = (name, frac)
#         # attach to candidate items
#         color_lookup = {ip: a_lookup.get(ip, ("",1.0)) for ip in items["image_path"]}

#         # anchor color if anchor row present
#         rel = str(anc_path.relative_to(img_root)).replace("\\","/") if str(anc_path).startswith(str(img_root)) else args.anchor
#         if rel in a_lookup:
#             anchor_color = a_lookup[rel][0]

#     # Compute anchor color if needed
#     _, anc_pil = load_image_tensor(anc_path, args.img_size)
#     if not anchor_color or args.anchor_color_mode in {"hsv","km"}:
#         if args.anchor_color_mode == "km":
#             anchor_color = dominant_color_kmeans(anc_pil, k=3, iters=8)
#         else:
#             # HSV median fallback
#             im = anc_pil.convert("RGB").resize((48,48), Image.BICUBIC)
#             arr = np.asarray(im, dtype=np.uint8)
#             hsv = np.zeros((arr.shape[0]*arr.shape[1], 3), dtype=np.float32)
#             k = 0
#             for y in range(arr.shape[0]):
#                 for x in range(arr.shape[1]):
#                     r,g,b = arr[y,x]
#                     h,s,v = rgb_to_hsv(int(r),int(g),int(b))
#                     hsv[k] = (h,s,v); k += 1
#             mask = hsv[:,1] >= 0.1
#             if mask.sum() == 0:
#                 vmean = hsv[:,2].mean()
#                 anchor_color = "white" if vmean>0.7 else ("black" if vmean<0.3 else "gray")
#             else:
#                 h_med = float(np.median(hsv[mask,0]))
#                 anchor_color = hsv_to_family(h_med, float(np.median(hsv[mask,1])), float(np.median(hsv[mask,2])))

#     # dataset/loader
#     ds = CatalogDS(items, img_root, img_size=args.img_size)
#     dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
#                     num_workers=max(0, args.num_workers), pin_memory=(device.type=="cuda"))

#     # encode anchor once
#     anc_tensor, _ = load_image_tensor(anc_path, args.img_size)
#     anc_tensor = anc_tensor.unsqueeze(0).to(device)
#     with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
#         fa = model.encoder(anc_tensor)

#     rows = []
#     for ids, rels, cats, titles, xb in tqdm(dl, desc="Scoring"):
#         xb = xb.to(device)
#         with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
#             fb = model.encoder(xb)
#             B = fb.shape[0]
#             logits = model.head(fa.repeat(B,1), fb)
#             probs  = torch.sigmoid(logits).detach().cpu().numpy()

#         # color bonus (weighted by dominance fraction if present)
#         if anchor_color and args.color_weight > 0:
#             rel_list = [str(r) for r in rels]
#             for i in range(B):
#                 cname, frac = color_lookup.get(rel_list[i], ("",1.0))
#                 bonus = args.color_weight * color_compat_score(anchor_color, cname) * max(0.5, float(frac or 0.0))
#                 probs[i] = float(probs[i]) + bonus

#         for i in range(B):
#             rows.append((str(ids[i]), str(rels[i]), str(cats[i]), str(titles[i]), float(probs[i])))

#     out = pd.DataFrame(rows, columns=["item_id","image_path","category","title","score"])
#     out["bucket"] = out["category"].apply(bucket_type)

#     # diversify by bucket
#     pieces = []
#     for b, grp in out.groupby("bucket"):
#         pieces.append(grp.sort_values("score", ascending=False).head(max(1, args.per_bucket)))
#     top = pd.concat(pieces, ignore_index=True).sort_values("score", ascending=False).head(args.topk).reset_index(drop=True)

#     Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
#     top.to_csv(args.out_csv, index=False)
#     print("✅ Wrote top-k ->", args.out_csv)
#     print("Anchor color (used):", anchor_color)
#     print("Top-10 preview:")
#     print(top.head(10).to_string(index=False))

# if __name__ == "__main__":
#     main()
# """, encoding="utf-8")
# print("Patched recommender: uses primary_* columns, optional k-means, dominance-weighted bonus")


# from __future__ import annotations
# import argparse, sys
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from PIL import Image, ImageFile
# from tqdm import tqdm

# import torch
# from torch.utils.data import DataLoader, Dataset

# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from src.models.compatibility import ImageOnlyCompatibility, IMAGENET_MEAN, IMAGENET_STD
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# def bucket_type(cat: str) -> str:
#     s = "" if cat is None else str(cat)
#     c = s.lower()
#     if any(k in c for k in ["sneaker","boot","heel","flat","sandal","loafer","shoe"]): return "footwear"
#     if any(k in c for k in ["jacket","coat","blazer","cardigan","hoodie","outer"]):    return "outerwear"
#     if any(k in c for k in ["dress","gown","jumper dress"]):                            return "dress"
#     if any(k in c for k in ["skirt","jeans","trouser","pant","short","legging","chino"]): return "bottoms"
#     if any(k in c for k in ["top","tee","t-shirt","shirt","blouse","sweater","pullover","tank","camisole"]): return "tops"
#     return "other"

# def load_image(path: Path, img_size: int):
#     im = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
#     arr = np.asarray(im, dtype=np.float32) / 255.0
#     arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
#     arr = np.transpose(arr, (2,0,1))
#     return torch.from_numpy(arr)

# class CatalogDS(Dataset):
#     def __init__(self, items_df: pd.DataFrame, images_root: Path, img_size: int):
#         self.df = items_df.reset_index(drop=True)
#         self.root = Path(images_root)
#         self.img_size = img_size
#     def __len__(self): return len(self.df)
#     def __getitem__(self, i):
#         r = self.df.iloc[i]
#         # force safe strings
#         item_id = "" if pd.isna(r.get("item_id")) else str(r.get("item_id"))
#         rel     = "" if pd.isna(r.get("image_path")) else str(r.get("image_path"))
#         cat     = "" if pd.isna(r.get("category")) else str(r.get("category"))
#         title   = "" if pd.isna(r.get("title")) else str(r.get("title"))
#         x = load_image(self.root / rel, self.img_size)
#         return item_id, rel, cat, title, x

# @torch.no_grad()
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", required=True)
#     ap.add_argument("--images-root", required=True)
#     ap.add_argument("--items-csv", required=True)
#     ap.add_argument("--anchor", required=True)
#     ap.add_argument("--img-size", type=int, default=224)
#     ap.add_argument("--topk", type=int, default=20)
#     ap.add_argument("--batch-size", type=int, default=256)
#     ap.add_argument("--num-workers", type=int, default=2)
#     ap.add_argument("--allow-types", default="", help="comma list: tops,bottoms,dress,outerwear,footwear")
#     ap.add_argument("--out-csv", default="outputs/eval/demo_recs.csv")
#     args = ap.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     ck = torch.load(args.ckpt, map_location=device)
#     model = ImageOnlyCompatibility(backbone=ck.get("backbone","efficientnet_b0"),
#                                    embed_dim=ck.get("embed_dim",512)).to(device)
#     model.load_state_dict(ck["model"]); model.eval()

#     img_root = Path(args.images_root)

#     # resolve anchor path (abs or relative to images-root)
#     anc_path = Path(args.anchor)
#     if not anc_path.exists():
#         anc_path = img_root / args.anchor
#     if not anc_path.exists():
#         raise SystemExit(f"❌ Anchor image not found: {args.anchor}")

#     # robust CSV load: make everything string-like, no NaNs
#     items = pd.read_csv(
#         args.items_csv,
#         dtype={"item_id": str, "image_path": str, "title": str, "category": str},
#         keep_default_na=False, na_values=[]
#     )
#     keep_cols = ["item_id","image_path","title","category"]
#     for c in keep_cols:
#         if c not in items.columns:
#             items[c] = ""
#     items[keep_cols] = items[keep_cols].fillna("")

#     # optional bucket filter
#     allow = set(s.strip().lower() for s in args.allow_types.split(",") if s.strip())
#     if allow:
#         items["bucket"] = items["category"].apply(bucket_type)
#         items = items[items["bucket"].isin(allow)].reset_index(drop=True)
#         if len(items) == 0:
#             raise SystemExit("❌ After filtering, catalog is empty. Relax --allow-types.")

#     # dataset/loader
#     ds = CatalogDS(items, img_root, img_size=args.img_size)
#     dl = DataLoader(
#         ds, batch_size=args.batch_size, shuffle=False,
#         num_workers=max(0, args.num_workers), pin_memory=(device.type=="cuda")
#     )

#     # encode anchor once
#     anc = load_image(anc_path, args.img_size).unsqueeze(0).to(device)
#     with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
#         fa = model.encoder(anc)  # [1, D]

#     rows = []
#     for batch in tqdm(dl, desc="Scoring"):
#         ids, rels, cats, titles, xb = batch
#         xb = xb.to(device)
#         with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
#             fb = model.encoder(xb)          # [B, D]
#             B = fb.shape[0]
#             logits = model.head(fa.repeat(B, 1), fb) # [B]
#             probs  = torch.sigmoid(logits).detach().cpu().numpy()
#         for i in range(B):
#             rows.append((str(ids[i]), str(rels[i]), str(cats[i]), str(titles[i]), float(probs[i])))

#     out = pd.DataFrame(rows, columns=["item_id","image_path","category","title","prob"])
#     out = out.sort_values("prob", ascending=False).reset_index(drop=True)
#     Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
#     out.to_csv(args.out_csv, index=False)
#     print("✅ Wrote full ranking ->", args.out_csv)
#     print("Top-10 preview:")
#     print(out.head(10).to_string(index=False))

# if __name__ == "__main__":
#     main()
