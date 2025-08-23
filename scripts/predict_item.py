# scripts/predict_item.py
import argparse, csv
from pathlib import Path
import sys

# >>> add repo root to import path so "src" is importable when running this file
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# <<<

from src.inference.item_classifier import ItemClassifier
from src.inference.image_utils import is_image_file

def iter_images(path: Path):
    if path.is_file() and is_image_file(path):
        yield path
    elif path.is_dir():
        for p in path.rglob("*"):
            if p.is_file() and is_image_file(p):
                yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models/item_classifier.yaml")
    ap.add_argument("--input", required=True, help="Image file or folder")
    ap.add_argument("--out-csv", default="outputs/predictions/item_preds.csv")
    args = ap.parse_args()

    clf = ItemClassifier(args.config)
    in_path = Path(args.input)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for img in iter_images(in_path):
        res = clf.predict_one(img)
        rows.append({
            "image": str(img),
            "pred_label": res["pred_label"],
            "pred_prob": f'{res["pred_prob"]:.4f}',
            "top1": f'{res["topk"][0]["label"]}:{res["topk"][0]["prob"]:.4f}',
            "top2": f'{res["topk"][1]["label"]}:{res["topk"][1]["prob"]:.4f}' if len(res["topk"])>1 else "",
            "top3": f'{res["topk"][2]["label"]}:{res["topk"][2]["prob"]:.4f}' if len(res["topk"])>2 else "",
        })

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"✅ Wrote predictions -> {out_path}  (rows: {len(rows)})")

if __name__ == "__main__":
    main()
