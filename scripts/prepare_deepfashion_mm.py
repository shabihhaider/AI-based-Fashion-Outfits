import argparse, json, re, shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def load_yaml(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)

def square_pad_resize(img: Image.Image, size: int) -> Image.Image:
    # letterbox pad to square, then resize
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BICUBIC)

def infer_category_from_path(path: Path, label_vocab: dict) -> str | None:
    tokens = re.split(r"[\\/ _\-\.]", path.as_posix().lower())
    # Flatten vocab to token->label mapping
    token_to_label = {}
    for group, cats in label_vocab["categories"].items():
        for c in cats:
            token_to_label[c] = c
    # simple heuristics
    for t in tokens[::-1]:  # prioritize deeper folder/file names
        if t in token_to_label:
            return token_to_label[t]
    # some common aliases -> canonical categories
    aliases = {
        "tee": "t_shirt", "tshirt": "t_shirt", "t-shirt": "t_shirt",
        "hooded": "hoodie", "sweatshirt": "sweater", "pullover": "sweater",
        "pants": "trousers", "slacks": "trousers", "denim": "jeans",
        "skorts": "skirt", "coatdress": "dress_midi", "jumper": "sweater",
        "parka": "parka", "windcheater": "windbreaker",
    }
    for t in tokens[::-1]:
        if t in aliases:
            return aliases[t]
    return None

def is_accessory_path(path: Path) -> bool:
    # skip bags, jewelry, hats, scarves, sunglasses, belts, wallets, ties, gloves
    accessory_words = {
        "bag","bags","handbag","backpack","wallet","belt","belts","scarf","scarves",
        "glasses","sunglasses","jewelry","necklace","ring","earring","bracelet",
        "hat","cap","beanie","tie","bowtie","gloves","watch"
    }
    tokens = re.split(r"[\\/ _\-\.]", path.as_posix().lower())
    return any(t in accessory_words for t in tokens)

def find_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            yield p

def try_load_deepfashion_annotations(raw_root: Path) -> dict:
    """
    Optional: attempt to load typical DeepFashion annotation files if present.
    Returns dict keyed by relative image path (lowercased) with fields.
    Supports a few common CSV/TXT layouts; silently skips if not found.
    """
    ann = {}

    # Helper to register a record
    def put(rel, **kwargs):
        key = rel.replace("\\", "/").lower()
        rec = ann.get(key, {})
        rec.update({k: v for k, v in kwargs.items() if v is not None})
        ann[key] = rec

    # Example: attributes.csv with columns: image_path,color,pattern,fabric,style,fit,season,category
    for fname in ["attributes.csv", "attributes.tsv", "annotations.csv"]:
        f = raw_root / fname
        if f.exists():
            sep = "," if f.suffix == ".csv" else "\t"
            try:
                df = pd.read_csv(f, sep=sep)
                # normalize column names
                df.columns = [c.strip().lower() for c in df.columns]
                path_col = None
                for candidate in ["image", "image_path", "path", "img", "file"]:
                    if candidate in df.columns:
                        path_col = candidate
                        break
                if path_col is None:
                    continue
                for _, r in df.iterrows():
                    rel = str(r[path_col]).strip()
                    put(
                        rel,
                        category_l2=r.get("category") if "category" in df.columns else None,
                        color_primary=r.get("color") or r.get("color_primary"),
                        color_secondary=r.get("color_secondary"),
                        pattern=r.get("pattern"),
                        fabric=r.get("fabric"),
                        style=r.get("style"),
                        fit=r.get("fit"),
                        sleeve_length=r.get("sleeve_length"),
                        neckline=r.get("neckline"),
                        length=r.get("length"),
                        season=r.get("season"),
                        bbox=r.get("bbox"),
                    )
            except Exception:
                pass

    # TXT formats (space or multi-space separated) like original DeepFashion lists
    for fname in ["list_category_img.txt", "list_attr_img.txt"]:
        f = raw_root / fname
        if f.exists():
            try:
                lines = [ln.strip() for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines()]
                # skip headers if any
                for ln in lines:
                    if not ln or ln.startswith("#") or " " not in ln:
                        continue
                    parts = re.split(r"\s+", ln)
                    rel = parts[0]
                    # last token could be a numeric id; skip unless known mapping
                    if len(parts) >= 2 and not parts[1].isdigit():
                        put(rel, category_l2=parts[1])
            except Exception:
                pass

    return ann

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--splits", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                        help="train/val/test fractions")
    parser.add_argument("--max-images", type=int, default=0, help="debug: cap total images")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    cfg_paths = load_yaml(ROOT / "configs" / "paths.yaml")
    cfg_labels = load_yaml(ROOT / "configs" / "labels.yaml")

    raw_root = Path(cfg_paths["raw"]["deepfashion_mm"])
    out_images = Path(cfg_paths["processed"]["deepfashion_mm"]["images"])
    out_manifests = Path(cfg_paths["processed"]["deepfashion_mm"]["manifests"])
    out_images.mkdir(parents=True, exist_ok=True)
    out_manifests.mkdir(parents=True, exist_ok=True)

    print(f"🔎 Scanning images under: {raw_root}")
    imgs = []
    for i, p in enumerate(find_images(raw_root)):
        if args.max_images and i >= args.max_images:
            break
        if is_accessory_path(p):
            continue
        imgs.append(p)
    if not imgs:
        raise SystemExit("No images found. Ensure dataset is extracted to data/raw/deepfashion_mm")

    # Attempt to load optional annotations
    ann = try_load_deepfashion_annotations(raw_root)

    rows = []
    for p in tqdm(imgs, desc="Indexing"):
        rel = safe_rel(p, raw_root).replace("\\", "/")
        base = {
            "src_path": str(p),
            "rel_path": rel,
            "category_l2": None,
            "color_primary": None,
            "color_secondary": None,
            "pattern": None,
            "fabric": None,
            "style": None,
            "fit": None,
            "sleeve_length": None,
            "neckline": None,
            "length": None,
            "season": None,
            "bbox": None,
        }
        # 1) From annotations (if any)
        if rel.lower() in ann:
            for k, v in ann[rel.lower()].items():
                base[k] = v if (pd.notna(v) and str(v).strip() != "") else None
        # 2) Heuristic from path
        if base["category_l2"] is None:
            cat = infer_category_from_path(p, cfg_labels)
            base["category_l2"] = cat
        rows.append(base)

    df = pd.DataFrame(rows)
    # Keep only rows that mapped to any clothing category (drop None)
    df = df[df["category_l2"].notna()].copy()
    if df.empty:
        raise SystemExit("No clothing images after filtering. Check labels.yaml or dataset layout.")

    # Stratified split by category
    train_frac, val_frac, test_frac = args.splits
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "splits must sum to 1.0"

    df_train, df_tmp = train_test_split(
        df, test_size=(1 - train_frac), stratify=df["category_l2"], random_state=42
    )
    rel = val_frac / (val_frac + test_frac) if (val_frac + test_frac) > 0 else 0.0
    df_val, df_test = train_test_split(
        df_tmp, test_size=(1 - rel), stratify=df_tmp["category_l2"], random_state=42
    )

    df_all = (
        pd.concat([
            df_train.assign(split="train"),
            df_val.assign(split="val"),
            df_test.assign(split="test"),
        ], axis=0)
        .reset_index(drop=True)
    )

    # Copy/resize to processed/images/{split}/...
    def copy_resize(subdf: pd.DataFrame, split: str):
        split_dir = out_images / split
        split_dir.mkdir(parents=True, exist_ok=True)
        out_rel_paths = []
        for _, r in tqdm(subdf.iterrows(), total=len(subdf), desc=f"Writing {split}"):
            src = Path(r["src_path"])
            # destination filename keeps partial folder context to reduce collisions
            dest_name = "_".join(Path(r["rel_path"]).parts[-3:])  # last up-to-3 segments
            dest_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", dest_name)
            if not dest_name.lower().endswith(".jpg"):
                dest_name = dest_name.rsplit(".", 1)[0] + ".jpg"
            dest = split_dir / dest_name

            try:
                with Image.open(src) as im:
                    out = square_pad_resize(im, args.img_size)
                    out.save(dest, format="JPEG", quality=90, optimize=True)
                out_rel_paths.append(str(dest.relative_to(out_images)).replace("\\", "/"))
            except Exception:
                # skip broken images
                continue
        return out_rel_paths

    train_out = copy_resize(df_all[df_all["split"] == "train"], "train")
    val_out   = copy_resize(df_all[df_all["split"] == "val"], "val")
    test_out  = copy_resize(df_all[df_all["split"] == "test"], "test")

    # Map back the new output relative paths (only for those successfully written)
    def rebuild_manifest(subdf, written_rel_paths, split):
        written_set = set(written_rel_paths)
        records = []
        for _, r in subdf.iterrows():
            # Reconstruct expected output name to check if it was written
            dest_name = "_".join(Path(r["rel_path"]).parts[-3:])
            dest_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", dest_name)
            if not dest_name.lower().endswith(".jpg"):
                dest_name = dest_name.rsplit(".", 1)[0] + ".jpg"
            rel = f"{split}/{dest_name}"
            if rel in written_set:
                rec = r.to_dict()
                rec["image_path"] = rel  # relative to processed/deepfashion_mm/images
                # Drop heavy fields
                rec.pop("src_path", None)
                rec.pop("rel_path", None)
                records.append(rec)
        return pd.DataFrame(records)

    mf_train = rebuild_manifest(df_all[df_all["split"] == "train"], train_out, "train")
    mf_val   = rebuild_manifest(df_all[df_all["split"] == "val"],   val_out,   "val")
    mf_test  = rebuild_manifest(df_all[df_all["split"] == "test"],  test_out,  "test")
    mf_all   = pd.concat([mf_train, mf_val, mf_test], axis=0).reset_index(drop=True)

    # Save manifests
    mf_all.to_csv(out_manifests / "all.csv", index=False)
    mf_train.to_csv(out_manifests / "train.csv", index=False)
    mf_val.to_csv(out_manifests / "val.csv", index=False)
    mf_test.to_csv(out_manifests / "test.csv", index=False)

    # Label stats
    stats = {
        "counts": mf_all["category_l2"].value_counts().to_dict(),
        "splits": {
            "train": len(mf_train),
            "val": len(mf_val),
            "test": len(mf_test),
            "total": len(mf_all),
        },
        "img_size": args.img_size,
    }
    (out_manifests / "label_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("✅ Done.")
    print("Manifests:", out_manifests)
    print("Images:", out_images)

if __name__ == "__main__":
    main()
