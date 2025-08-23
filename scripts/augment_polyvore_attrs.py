# augment_polyvore_attrs.py
import argparse, sys, json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

# allow 'src' imports when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.clip_attrs import ZeroShotAttrModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_paths_yaml():
    import yaml
    return yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))

def load_items_manifest(manif_root: Path, split: str) -> Path:
    # prefer color-augmented items if present
    aug = manif_root / f"items_{split}_aug.csv"
    base = manif_root / f"items_{split}.csv"
    if aug.exists():
        return aug
    elif base.exists():
        return base
    else:
        raise SystemExit(f"❌ Missing items manifest for split '{split}'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train","val","test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--model-name", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    args = ap.parse_args()

    cfg = read_paths_yaml()
    images_root = Path(cfg["processed"]["polyvore"]["images"])
    manif_root  = Path(cfg["processed"]["polyvore"]["manifests"])

    zs = ZeroShotAttrModel(device=args.device, model_name=args.model_name, pretrained=args.pretrained)

    for split in args.splits:
        items_csv = load_items_manifest(manif_root, split)
        df = pd.read_csv(items_csv)

        if "image_path" not in df.columns:
            raise SystemExit(f"❌ {items_csv} missing 'image_path' column")

        rows = []
        B = args.batch_size
        paths = df["image_path"].tolist()

        for i in tqdm(range(0, len(paths), B), desc=f"Attrs [{split}]"):
            batch_paths = paths[i:i+B]
            ims = []
            for p in batch_paths:
                try:
                    im = Image.open(images_root / p).convert("RGB")
                except Exception:
                    im = None
                ims.append(im)

            # Replace failed images with a white placeholder to keep batch shape
            safe_ims = [im if im is not None else Image.new("RGB", (224,224), (255,255,255)) for im in ims]
            scores = zs.predict_batch(safe_ims)

            # decode top-1 & top-3 strings
            for bi, rel in enumerate(batch_paths):
                out = {"image_path": rel}
                for attr in ["pattern","fabric","style","fit"]:
                    top3 = zs.decode_topk(scores[attr][bi:bi+1, :], attr, k=3)[0]
                    out[f"{attr}_top1"] = top3[0][0]
                    out[f"{attr}_p_top1"] = round(top3[0][1], 4)
                    out[f"{attr}_top3"] = "; ".join([f"{lab}:{prob:.3f}" for lab, prob in top3])
                rows.append(out)

        out_csv = manif_root / f"items_{split}_attr.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"✅ Wrote {out_csv}  (rows: {len(rows)})")

if __name__ == "__main__":
    main()
