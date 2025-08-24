# scripts/build_feats_cache.py
from pathlib import Path
import argparse, torch, pandas as pd
from tqdm import tqdm
from src.service.recommender import RecommendEngine, OptimizedCatalogDS, load_image_preproc

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    eng = RecommendEngine(
        ckpt_path=Path(args.ckpt),
        catalog_csv=Path(args.catalog),
        images_root=Path(args.images_root),
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        items_aug_csv=None,
        item_cls_onnx=None, item_cls_meta=None,
        topn_rerank=0,
    )

    ds = OptimizedCatalogDS(eng.items, eng.images_root, eng.img_size)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    feats, paths = [], []
    for _, batch in enumerate(tqdm(dl, desc="Catalog feats")):
        ids, rels, cats, titles, xb = batch
        xb = xb.to(eng.device)
        with torch.amp.autocast(device_type=eng.device.type, enabled=(eng.device.type=="cuda")):
            fb = eng.model.encoder(xb).detach().cpu()
        feats.append(fb)
        paths.extend(rels)

    feats = torch.cat(feats, dim=0).contiguous()
    obj = {
        "feats": feats,          # [N, D]
        "paths": list(paths),    # image_path list aligned to eng.items
        "embed_dim": feats.shape[1],
        "img_size": eng.img_size,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, args.out)
    print(f"✅ wrote cache: {args.out}  shape={tuple(feats.shape)}")

if __name__ == "__main__":
    main()
