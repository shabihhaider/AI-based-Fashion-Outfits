import argparse, json, random, re, csv
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
import yaml
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(42)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ------------------------- utils -------------------------
def load_yaml(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_category_map(root: Path):
    """
    Reads categories.csv -> maps numeric id to readable name.
    Handles common column variants.
    """
    mp = {}
    f = root / "categories.csv"
    if f.exists():
        df = pd.read_csv(f)
        cols = {c.lower(): c for c in df.columns}
        id_col = cols.get("id") or cols.get("category_id") or list(df.columns)[0]
        name_col = cols.get("name") or cols.get("category") or cols.get("category_name") or list(df.columns)[-1]
        for _, r in df.iterrows():
            try:
                mp[str(int(r[id_col]))] = str(r[name_col])
            except Exception:
                mp[str(r[id_col])] = str(r[name_col])
    return mp


def load_item_metadata(root: Path):
    """
    Loads polyvore_item_metadata.json.
    Returns dict[item_id] = { 'img': <filename or path>, 'title': <str>, 'category': <id or name> }
    Robust to list/dict shapes and key name variants.
    """
    meta_file = root / "polyvore_item_metadata.json"
    if not meta_file.exists():
        return {}
    raw = json.loads(meta_file.read_text(encoding="utf-8"))
    items = {}

    def add_one(rec, forced_id=None):
        iid = str(forced_id or rec.get("item_id") or rec.get("id") or rec.get("itemId") or "")
        if not iid:
            return
        img = (
            rec.get("image")
            or rec.get("img")
            or rec.get("image_path")
            or rec.get("filename")
            or rec.get("url")
            or ""
        )
        title = rec.get("title") or rec.get("name") or rec.get("desc") or ""
        cat = (
            rec.get("category")
            or rec.get("category_name")
            or rec.get("semantic_category")
            or rec.get("categoryid")
            or rec.get("category_id")
        )
        items[iid] = {"img": str(img), "title": str(title), "category": cat}

    if isinstance(raw, list):
        for rec in raw:
            add_one(rec)
    elif isinstance(raw, dict):
        if "items" in raw and isinstance(raw["items"], list):
            for rec in raw["items"]:
                add_one(rec)
        else:
            # dict keyed by item_id
            for k, v in raw.items():
                add_one(v, forced_id=k)

    return items


def load_outfits_from_split(root: Path, variant: str, split: str):
    """
    Reads disjoint/nondisjoint/<split>.json and returns list[list_of_item_ids].
    """
    fname = "valid.json" if split == "val" else f"{split}.json"
    jpath = root / variant / fname
    if not jpath.exists():
        return []

    data = json.loads(jpath.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "outfits" in data:
        data = data["outfits"]

    outfits = []
    for o in data:
        items = o.get("items") or o.get("item_ids") or o.get("itemIdList") or []
        ids = []
        for it in items:
            if isinstance(it, dict):
                ids.append(str(it.get("item_id") or it.get("id") or it.get("itemId") or "").strip())
            else:
                ids.append(str(it).strip())
        ids = [i for i in ids if i]
        if len(ids) >= 2:
            outfits.append(ids)
    return outfits


def is_accessory(text: str) -> bool:
    acc = {
        "bag",
        "bags",
        "handbag",
        "backpack",
        "wallet",
        "belt",
        "belts",
        "scarf",
        "scarves",
        "glasses",
        "sunglasses",
        "jewelry",
        "necklace",
        "ring",
        "earring",
        "bracelet",
        "hat",
        "cap",
        "beanie",
        "tie",
        "bowtie",
        "gloves",
        "watch",
        "socks",
        "stockings",
    }
    tokens = re.split(r"[\\/ _\-\.\,]+", text.lower())
    return any(t in acc for t in tokens)


def square_pad_resize(img: Image.Image, size: int) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resolve_image_path(img_field: str, raw_root: Path, iid: str | None = None):
    """
    Try several locations:
      - raw_root / <img_field> (if it's already a path)
      - raw_root / images / <basename>
      - raw_root / images / <iid>.jpg|.png|.jpeg (fallback by item_id)
    """
    cand = raw_root / img_field
    if cand.exists():
        return cand

    base = Path(img_field).name
    cand = raw_root / "images" / base
    if cand.exists():
        return cand

    if iid:
        for ext in (".jpg", ".png", ".jpeg", ".webp"):
            c = raw_root / "images" / f"{iid}{ext}"
            if c.exists():
                return c

    # last resort: basename search
    hits = list((raw_root / "images").rglob(base))
    return hits[0] if hits else None


def write_resized_images(items, raw_root: Path, out_images_root: Path, img_size: int):
    written = {}
    for iid, meta in tqdm(items.items(), desc="Writing images"):
        src = resolve_image_path(meta["img"], raw_root, iid)
        if not src or not src.suffix.lower() in IMG_EXTS:
            continue
        split_dir = out_images_root / meta["split"]
        ensure_dir(split_dir)
        # filename: <itemid>_<basename>.jpg
        base = Path(meta["img"]).name
        base = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base)
        if not base.lower().endswith(".jpg"):
            base = base.rsplit(".", 1)[0] + ".jpg"
        dest = split_dir / f"{iid}_{base}"
        try:
            with Image.open(src) as im:
                out = square_pad_resize(im, img_size)
                out.save(dest, format="JPEG", quality=90, optimize=True)
            written[iid] = str(dest.relative_to(out_images_root)).replace("\\", "/")
        except Exception:
            continue
    return written


def build_existing_map(out_images_root: Path):
    """
    Rebuild {item_id -> split/filename.jpg} from already-written files.
    Assumes filenames like '<itemid>_<basename>.jpg'.
    """
    mapping = {}
    for split in ["train", "val", "test"]:
        split_dir = out_images_root / split
        if not split_dir.exists():
            continue
        for p in split_dir.glob("*.jpg"):
            name = p.name
            if "_" in name:
                iid = name.split("_", 1)[0]
                mapping[iid] = f"{split}/{name}"
    return mapping


def write_pairs_stream(outfits_by_split, ok_items, out_dir: Path, neg_per_pos: int):
    """
    Stream positives and sampled negatives directly to CSV per split.
    """
    for split in ["train", "val", "test"]:
        outfits = outfits_by_split.get(split, [])
        if not outfits:
            continue

        # Build pool of available items in this split
        pool = sorted({iid for lst in outfits for iid in lst if iid in ok_items})
        pos_set = set()  # to avoid duplicate positives across outfits

        out_csv = out_dir / f"pairs_{split}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["img_a", "img_b", "label"])

            pbar = tqdm(outfits, desc=f"Writing pairs [{split}]", unit="outfit")
            for lst in pbar:
                # positives: all 2-combinations
                n = len(lst)
                for i in range(n):
                    a = lst[i]
                    if a not in ok_items:
                        continue
                    for j in range(i + 1, n):
                        b = lst[j]
                        if b not in ok_items:
                            continue
                        key = (a, b) if a < b else (b, a)
                        if key in pos_set:
                            continue
                        pos_set.add(key)
                        w.writerow([ok_items[a]["image_path"], ok_items[b]["image_path"], 1])

                        # negatives: sample k per positive
                        for _ in range(neg_per_pos):
                            tries = 0
                            while tries < 10:
                                x = random.choice(pool)
                                if x == a or x == b:
                                    tries += 1
                                    continue
                                k = (a, x) if a < x else (x, a)
                                if k not in pos_set and x in ok_items:
                                    w.writerow([ok_items[a]["image_path"], ok_items[x]["image_path"], 0])
                                    break
                                tries += 1


def write_triplets_stream(outfits_by_split, ok_items, out_dir: Path, pos_set_by_split):
    """
    Stream triplets: for each positive (a,b) pick a c s.t. (a,c) is NOT positive.
    """
    for split in ["train", "val", "test"]:
        outfits = outfits_by_split.get(split, [])
        if not outfits:
            continue
        pool = sorted({iid for lst in outfits for iid in lst if iid in ok_items})
        pos_set = pos_set_by_split.get(split, set())

        out_csv = out_dir / f"triplets_{split}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["anchor", "positive", "negative"])

            pbar = tqdm(outfits, desc=f"Writing triplets [{split}]", unit="outfit")
            for lst in pbar:
                n = len(lst)
                for i in range(n):
                    a = lst[i]
                    if a not in ok_items:
                        continue
                    for j in range(i + 1, n):
                        b = lst[j]
                        if b not in ok_items:
                            continue
                        key = (a, b) if a < b else (b, a)
                        if key not in pos_set:
                            pos_set.add(key)
                        # sample a hard negative c
                        tries = 0
                        while tries < 20:
                            c = random.choice(pool)
                            if c == a or c == b:
                                tries += 1
                                continue
                            k = (a, c) if a < c else (c, a)
                            if k not in pos_set:
                                w.writerow(
                                    [
                                        ok_items[a]["image_path"],
                                        ok_items[b]["image_path"],
                                        ok_items[c]["image_path"],
                                    ]
                                )
                                break
                            tries += 1


def count_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8") as f:
        # subtract header
        return max(sum(1 for _ in f) - 1, 0)


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--make-triplets", action="store_true")
    ap.add_argument(
        "--variant",
        choices=["disjoint", "nondisjoint"],
        default="disjoint",
        help="Choose split set from the dataset root (default: disjoint)",
    )
    ap.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip resizing/copy if images already exist in processed/ (much faster)",
    )
    ap.add_argument(
        "--neg-per-pos",
        type=int,
        default=1,
        help="Number of negatives to write per positive pair (default: 1)",
    )
    ap.add_argument(
        "--max-outfits",
        type=int,
        default=0,
        help="Debug cap on outfits per split (0 = no cap)",
    )
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    cfg_paths = load_yaml(ROOT / "configs" / "paths.yaml")

    raw_root = Path(cfg_paths["raw"]["polyvore_outfit"])
    out_images = Path(cfg_paths["processed"]["polyvore"]["images"])
    out_manifests = Path(cfg_paths["processed"]["polyvore"]["manifests"])
    ensure_dir(out_images)
    ensure_dir(out_manifests)

    variant = args.variant  # "disjoint" or "nondisjoint"

    # Load maps & metadata
    cat_map = load_category_map(raw_root)
    meta = load_item_metadata(raw_root)

    # 1) collect outfits and items, filter accessories
    all_items = {}
    all_outfits = []
    for split in ["train", "val", "test"]:
        outfit_lists = load_outfits_from_split(raw_root, variant, split)
        for lst in outfit_lists:
            kept = []
            for iid in lst:
                m = meta.get(iid, {})
                title = m.get("title", "")
                cat = m.get("category", "")
                if isinstance(cat, (int, float)) or (isinstance(cat, str) and cat.isdigit()):
                    cat_name = cat_map.get(str(int(cat))) or cat_map.get(str(cat)) or ""
                else:
                    cat_name = str(cat)

                # filter out accessories
                text_for_filter = f"{title} {cat_name} {m.get('img','')}"
                if is_accessory(text_for_filter):
                    continue

                img_field = m.get("img") or f"{iid}.jpg"
                if iid not in all_items:
                    all_items[iid] = {
                        "img": img_field,
                        "title": title,
                        "category": cat_name,
                        "split": split,
                    }
                kept.append(iid)
            if len(kept) >= 2:
                all_outfits.append((split, kept))

    if not all_items or not all_outfits:
        raise SystemExit(
            "❌ Parsed zero items/outfits. Check that disjoint/nondisjoint JSONs and metadata exist."
        )

    # Group outfits by split and optionally cap
    outfits_by_split = {"train": [], "val": [], "test": []}
    for split, lst in all_outfits:
        outfits_by_split[split].append(lst)
    if args.max_outfits > 0:
        for s in outfits_by_split:
            outfits_by_split[s] = outfits_by_split[s][: args.max_outfits]

    # 2) write or reuse images
    if args.skip_images:
        print("⏭️  Skipping image resize/copy; rebuilding map from processed folder...")
        written_rel = build_existing_map(out_images)
    else:
        written_rel = write_resized_images(all_items, raw_root, out_images, args.img_size)

    # 3) rebuild outfits with surviving items & ok_items mapping
    filtered = {"train": [], "val": [], "test": []}
    ok_items = {}
    for split, lists in outfits_by_split.items():
        for lst in lists:
            kept = [iid for iid in lst if iid in written_rel]
            if len(kept) >= 2:
                filtered[split].append(kept)
                for iid in kept:
                    if iid not in ok_items:
                        ok_items[iid] = {"image_path": written_rel[iid], "split": split}

    if not any(filtered.values()):
        raise SystemExit("❌ After filtering, no outfits have 2+ items with images. Check dataset extraction.")

    # 4) items_<split>.csv
    by_split_items = {"train": [], "val": [], "test": []}
    for iid, info in ok_items.items():
        s = info["split"]
        m = all_items.get(iid, {})
        by_split_items[s].append(
            {
                "item_id": iid,
                "image_path": info["image_path"],
                "title": m.get("title", ""),
                "category": m.get("category", ""),
            }
        )
    for split, rows in by_split_items.items():
        if rows:
            pd.DataFrame(rows).to_csv(out_manifests / f"items_{split}.csv", index=False)

    # 5) stream pairs
    write_pairs_stream(filtered, ok_items, out_manifests, args.neg_per_pos)

    # 6) build positive sets for triplets
    pos_set_by_split = {}
    for split, lists in filtered.items():
        s = set()
        for lst in lists:
            n = len(lst)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = lst[i], lst[j]
                    key = (a, b) if a < b else (b, a)
                    s.add(key)
        pos_set_by_split[split] = s

    # 7) stream triplets if requested
    if args.make_triplets:
        write_triplets_stream(filtered, ok_items, out_manifests, pos_set_by_split)

    # 8) stats
    stats = {
        "items_per_split": {k: len(v) for k, v in by_split_items.items()},
        "pairs": {
            s: count_rows(out_manifests / f"pairs_{s}.csv") for s in ["train", "val", "test"]
        },
    }
    if args.make_triplets:
        stats["triplets"] = {
            s: count_rows(out_manifests / f"triplets_{s}.csv") for s in ["train", "val", "test"]
        }
    (out_manifests / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("✅ Polyvore preprocessing complete.")
    print("Manifests:", out_manifests)
    print("Images:", out_images)


if __name__ == "__main__":
    main()
