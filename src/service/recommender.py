# src/service/recommender.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch.utils.data import DataLoader, Dataset

import colorsys

def _hex_to_hue(hexstr: str | None):
    if not hexstr or not isinstance(hexstr, str): return None
    try:
        h = hexstr.lstrip("#")
        r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return colorsys.rgb_to_hsv(r/255, g/255, b/255)[0] * 360.0
    except Exception:
        return None

def _hue_bonus(anchor_h, cand_h, mode: str):
    if anchor_h is None or cand_h is None: return 0.0
    d = abs(anchor_h - cand_h); d = min(d, 360 - d)
    if mode == "match":        # prefer same color
        return max(0.0, 1.0 - d/60.0)
    if mode == "analogous":    # ±30°
        return max(0.0, 1.0 - abs(d-30)/60.0)
    if mode == "neutral":      # prefer gray/white/black
        return 0.0  # (we don’t compute grayness here)
    # default: complement (~180°)
    return max(0.0, 1.0 - abs(d-180)/90.0)

# ---- tiny utils reused from your training code ----
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
ImageFile.LOAD_TRUNCATED_IMAGES = True

NEUTRALS = {"black","white","gray","grey","beige","cream","denim","navy","tan","brown"}

def _extract_state_dict(obj):
    """
    Accepts anything from torch.save:
      - raw state_dict (keys like 'encoder.*', 'head.*')
      - {'model': state_dict, 'meta': {...}}
      - {'state_dict': state_dict, ...}
      - {'model_state_dict': state_dict, ...}
    Also strips common prefixes: 'module.' and 'model.'.
    """
    sd = None
    if isinstance(obj, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break
        if sd is None:
            # sometimes the dict itself is the state dict
            # (keys look like 'encoder.conv_stem.weight', etc.)
            looks_like_sd = any(isinstance(v, torch.Tensor) for v in obj.values())
            if looks_like_sd:
                sd = obj
    else:
        sd = obj

    if sd is None:
        raise RuntimeError("Could not locate a state_dict in checkpoint")

    # strip prefixes
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        if nk.startswith("model."):
            nk = nk[6:]
        new_sd[nk] = v
    return new_sd

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
    return {"grey":"gray", "navy blue":"navy"}.get(n, n)

def hex_to_rgb(hexstr: str):
    if not isinstance(hexstr, str) or not hexstr: return None
    s = hexstr.strip().lstrip("#")
    if len(s) == 3: s = "".join([ch*2 for ch in s])
    if len(s) != 6: return None
    try: return (int(s[0:2],16), int(s[2:4],16), int(s[4:6],16))
    except Exception: return None

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

def color_compat_score(anchor_color: str, cand_color: str) -> float:
    a = normalize_color_name(anchor_color)
    b = normalize_color_name(cand_color)
    if not a or not b: return 0.0
    if a == b: return 0.5
    if a in NEUTRALS or b in NEUTRALS: return 0.3
    # simple complementary ideas
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
    if a in COMPLEMENT and b in COMPLEMENT[a]: return 0.4
    return 0.0

def load_image_preproc(img: Image.Image, img_size: int) -> torch.Tensor:
    im = img.convert("RGB").resize((img_size, img_size), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    arr = np.transpose(arr, (2,0,1))
    return torch.from_numpy(arr)

# ---- model wrapper ----
class ImageOnlyCompatibility(torch.nn.Module):
    """Minimal head wrapper to load your checkpoint.
    Assumes encoder backbone + a pairwise head that maps (fa, fb) -> logit."""
    def __init__(self, backbone="efficientnet_b0", embed_dim=512):
        super().__init__()
        import timm
        self.encoder = timm.create_model(backbone, pretrained=False, num_classes=embed_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(embed_dim*2, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1),
        )
    def forward(self, xa, xb):
        fa = self.encoder(xa)
        fb = self.encoder(xb)
        x  = torch.cat([fa, fb], dim=1)
        return self.head(x).squeeze(1)

class CatalogDS(Dataset):
    def __init__(self, items_df: pd.DataFrame, images_root: Path, img_size: int):
        self.df = items_df.reset_index(drop=True)
        self.root = Path(images_root)
        self.img_size = img_size
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = Image.open(self.root / r["image_path"]).convert("RGB")
        x = load_image_preproc(x, self.img_size)
        return str(r["item_id"]), str(r["image_path"]), str(r["category"]), str(r["title"]), x

class RecommendEngine:
    def __init__(
        self,
        ckpt_path: Path,
        catalog_csv: Path,
        images_root: Path,
        items_aug_csv: Optional[Path] = None,
        img_size: int = 224,
        batch_size: int = 256,
        num_workers: int = 0,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = Path(ckpt_path)
        self._feat_cache = {}  # sha1 -> torch.Tensor on CPU
        # 1) load checkpoint
        ck = torch.load(self.ckpt_path, map_location="cpu")
        meta = ck.get("meta", {}) if isinstance(ck, dict) else {}

        # 2) choose backbone & embed size (from meta if present)
        backbone = (
            meta.get("arch")
            or (ck.get("backbone") if isinstance(ck, dict) else None)
            or "efficientnet_b0"
        )
        embed_dim = (
            meta.get("embed_dim")
            or (ck.get("embed_dim") if isinstance(ck, dict) else None)
            or 512
        )

        # 3) build model FIRST, then load weights
        self.model = ImageOnlyCompatibility(backbone=backbone, embed_dim=embed_dim).to(self.device)

        sd = _extract_state_dict(ck)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)

        # (optional) adopt image size from meta, else keep arg
        self.img_size = int(meta.get("img_size", img_size))

        # helpful logs
        if missing:
            print(f"[compat] missing keys: {len(missing)} (e.g., {missing[:5]})")
        if unexpected:
            print(f"[compat] unexpected keys: {len(unexpected)} (e.g., {unexpected[:5]})")

        self.model.eval()

        # catalog
        items = pd.read_csv(
            catalog_csv,
            dtype={"item_id": str, "image_path": str, "title": str, "category": str},
            keep_default_na=False, na_values=[]
        ).fillna("")
        for c in ["item_id","image_path","title","category"]:
            if c not in items.columns: items[c] = ""
        items["bucket"] = items["category"].apply(bucket_type)
        self.items = items
        self.images_root = Path(images_root)
        self.batch_size = batch_size
        self.num_workers = max(0, num_workers)

        # color aug lookup
        self.color_lookup: Dict[str, Tuple[str,float]] = {}
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
            self.color_lookup = {ip: tmp.get(ip, ("",1.0)) for ip in self.items["image_path"]}

    def _ensure_clip_bucket(self):
        if getattr(self, "_clip_bucket", None) is not None:
            return
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            tok = open_clip.get_tokenizer("ViT-B-32")
            self._clip_bucket = {"model": model.eval().to(self.device), "prep": preprocess, "tok": tok}
            self._clip_prompts = {
                "tops": ["a blouse", "a shirt", "a t-shirt", "a sweater"],
                "bottoms": ["jeans", "trousers", "pants", "a skirt", "shorts"],
                "outerwear": ["a jacket", "a coat", "a blazer", "a cardigan"],
                "footwear": ["shoes", "boots", "heels", "sneakers"],
            }
        except Exception:
            self._clip_bucket = None

    @torch.no_grad()
    def predict_anchor_bucket(self, pil: Image.Image) -> str:
        self._ensure_clip_bucket()
        if self._clip_bucket is None:
            return "unknown"
        clip = self._clip_bucket
        prompts = sum(self._clip_prompts.values(), [])
        img = clip["prep"](pil).unsqueeze(0).to(self.device)
        txt = clip["tok"](prompts).to(self.device)
        with torch.amp.autocast('cuda', enabled=(self.device.type=="cuda")):
            im = clip["model"].encode_image(img)
            tx = clip["model"].encode_text(txt)
            sim = (im @ tx.T).softmax(dim=-1)[0].float().cpu().numpy()
        i = 0; best=("unknown",-1.0)
        for b, lst in self._clip_prompts.items():
            s = float(sim[i:i+len(lst)].sum()); i += len(lst)
            if s > best[1]: best = (b, s)
        return best[0]

    @torch.no_grad()
    def recommend(
        self,
        user_image: Image.Image,
        *,
        allow_types: str = "tops,bottoms,outerwear,footwear",
        per_bucket: int = 5,
        topk: int = 20,
        color_weight: float = 0.1,
        anchor_color_mode: str = "auto",   # "auto"|"hsv"|"km"
    ) -> Tuple[str, pd.DataFrame]:
        # anchor color
        if anchor_color_mode == "km":
            # minimal k-means
            im = user_image.convert("RGB").resize((64,64), Image.BICUBIC)
            X = np.asarray(im, dtype=np.float32).reshape(-1,3)
            idx = np.random.choice(X.shape[0], size=min(3, X.shape[0]), replace=False)
            C = X[idx].copy()
            for _ in range(8):
                d2 = ((X[:,None,:]-C[None,:,:])**2).sum(axis=2)
                lab = d2.argmin(axis=1)
                for j in range(C.shape[0]):
                    pts = X[lab==j]
                    if len(pts): C[j] = pts.mean(axis=0)
            counts = np.bincount(lab, minlength=C.shape[0])
            j = int(counts.argmax())
            r,g,b = C[j]
            h,s,v = rgb_to_hsv(int(r),int(g),int(b))
            anchor_color = hsv_to_family(h,s,v)
        else:
            anchor_color = dominant_color_hsv_fast(user_image)

        # filter catalog if requested
        allow = set(s.strip().lower() for s in allow_types.split(",") if s.strip())
        items = self.items
        if allow:
            items = items[items["bucket"].isin(allow)].reset_index(drop=True)
            if len(items) == 0:
                return anchor_color, items

        # encode anchor once
        xa = load_image_preproc(user_image, self.img_size).unsqueeze(0).to(self.device)
        with torch.amp.autocast('cuda', enabled=(self.device.type=="cuda")):
            fa = self.model.encoder(xa)

        # score catalog
        ds = CatalogDS(items, self.images_root, self.img_size)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                        num_workers=self.num_workers, pin_memory=(self.device.type=="cuda"))

        rows: List[Tuple[str,str,str,str,float]] = []
        for ids, rels, cats, titles, xb in dl:
            xb = xb.to(self.device)
            with torch.amp.autocast('cuda', enabled=(self.device.type=="cuda")):
                fb = self.model.encoder(xb)
                B = fb.shape[0]
                logits = self.model.head(torch.cat([fa.repeat(B,1), fb], dim=1))
                probs  = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()

            if self.color_lookup and color_weight > 0:
                for i in range(B):
                    cname, frac = self.color_lookup.get(str(rels[i]), ("",1.0))
                    bonus = color_weight * color_compat_score(anchor_color, cname) * max(0.5, float(frac or 0.0))
                    probs[i] = float(probs[i]) + bonus

            for i in range(B):
                rows.append((str(ids[i]), str(rels[i]), str(cats[i]), str(titles[i]), float(probs[i])))

        out = pd.DataFrame(rows, columns=["item_id","image_path","category","title","score"])
        out["bucket"] = out["category"].apply(bucket_type)
        
        # ---------- 3a) Infer anchor bucket and DROP same-bucket items ----------
        try:
            anchor_bucket = self.predict_anchor_bucket(user_image)
        except Exception:
            anchor_bucket = "unknown"

        if anchor_bucket in {"tops", "bottoms", "outerwear", "footwear"}:
            out = out[out["bucket"] != anchor_bucket].copy()

        # ---------- 3b) Hue-based complementary bonus (optional, uses items_aug) ----------
        if float(color_weight) > 0:
            # quick anchor hue from the uploaded image
            try:
                arr = np.asarray(user_image.convert("RGB").resize((64, 64)), dtype=np.uint8).reshape(-1, 3)
                r_med, g_med, b_med = np.median(arr[:, 0]), np.median(arr[:, 1]), np.median(arr[:, 2])
                anchor_h = colorsys.rgb_to_hsv(r_med/255.0, g_med/255.0, b_med/255.0)[0] * 360.0
            except Exception:
                anchor_h = None

            # ensure candidate hex available
            if "primary_hex" not in out.columns and getattr(self, "items_aug", None) is not None:
                out = out.merge(self.items_aug[["image_path", "primary_hex"]], on="image_path", how="left")

            if "primary_hex" in out.columns:
                # compute per-row hue bonus and add to score
                def _bonus_row(row):
                    cand_h = _hex_to_hue(row.get("primary_hex"))
                    return _hue_bonus(anchor_h, cand_h, mode="complement")
                out["score"] = out["score"].astype(float) + float(color_weight) * out.apply(_bonus_row, axis=1)

        # diversify per bucket
        pieces = []
        for b, grp in out.groupby("bucket"):
            pieces.append(grp.sort_values("score", ascending=False).head(max(1, per_bucket)))
        top = pd.concat(pieces, ignore_index=True).sort_values("score", ascending=False).head(topk).reset_index(drop=True)
        return anchor_color, top
