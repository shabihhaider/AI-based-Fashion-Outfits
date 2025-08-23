# augment_polyvore_colors.py
import argparse, sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import yaml
from PIL import Image

# make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.colors import extract_colors_from_image

def load_yaml(p: Path):
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def process_split(items_csv: Path, images_root: Path, out_csv: Path):
    df = pd.read_csv(items_csv)
    if "image_path" not in df.columns:
        raise SystemExit(f"❌ {items_csv} missing 'image_path' column.")
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Colors [{items_csv.stem}]"):
        img_rel = r["image_path"]
        img_abs = images_root / img_rel
        try:
            with Image.open(img_abs) as im:
                meta = extract_colors_from_image(im)
        except Exception:
            meta = {k: "" for k in ["primary_name","primary_hex","primary_frac","secondary_name","secondary_hex","secondary_frac","brightness","saturation"]}
        rows.append({**r.to_dict(), **meta})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv}  (rows: {len(rows)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train","val","test"], help="Which splits to process")
    args = ap.parse_args()

    cfg = load_yaml(ROOT / "configs" / "paths.yaml")
    images_root = Path(cfg["processed"]["polyvore"]["images"])
    mani_root   = Path(cfg["processed"]["polyvore"]["manifests"])

    for s in args.splits:
        items_csv = mani_root / f"items_{s}.csv"
        out_csv   = mani_root / f"items_{s}_aug.csv"
        if not items_csv.exists():
            print(f"⚠️  Skip missing {items_csv}")
            continue
        process_split(items_csv, images_root, out_csv)

if __name__ == "__main__":
    main()
