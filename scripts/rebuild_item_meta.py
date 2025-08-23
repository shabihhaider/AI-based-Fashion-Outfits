# scripts/rebuild_item_meta.py
import argparse, json, zipfile
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help=r"Folder containing item_classifier_best.pt and item_classifier.onnx")
    ap.add_argument("--pt", default="item_classifier_best.pt")
    ap.add_argument("--onnx", default="item_classifier.onnx")
    ap.add_argument("--make-zip", action="store_true", help="Create item_classifier_artifacts.zip")
    args = ap.parse_args()

    root = Path(args.root)
    pt_path = root / args.pt
    onnx_path = root / args.onnx

    if not pt_path.exists():
        raise SystemExit(f"❌ Missing checkpoint: {pt_path}")
    if not onnx_path.exists():
        print(f"⚠️  ONNX not found at {onnx_path} (continuing to write meta.json)")

    # rebuild meta from checkpoint
    ckpt = torch.load(pt_path, map_location="cpu")
    meta = {
        "arch": "efficientnet_b0",
        "img_size": ckpt.get("img_size", 512),
        "classes": ckpt.get("classes", []),
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    meta_path = root / "item_classifier_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"✅ Wrote {meta_path}")

    # optional: zip all artifacts
    if args.make_zip:
        zip_path = root / "item_classifier_artifacts.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(pt_path, arcname=pt_path.name)
            if onnx_path.exists():
                z.write(onnx_path, arcname=onnx_path.name)
            z.write(meta_path, arcname=meta_path.name)
        print(f"📦 Zipped -> {zip_path}")

if __name__ == "__main__":
    main()
