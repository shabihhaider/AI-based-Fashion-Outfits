# app/main.py
import os
import sys
from pathlib import Path
from typing import List, Annotated

import torch
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------------- project setup ----------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service.recommender import RecommendEngine, load_image_preproc

# ---------------- env/config ----------------
CKPT    = Path(os.getenv("OUTFIT_CKPT",    str(ROOT / "models" / "compatibility_best.pt")))
CATCSV  = Path(os.getenv("OUTFIT_CATALOG", str(ROOT / "data/processed/polyvore/manifests_disjoint/items_test.csv")))
AUGCSV  = Path(os.getenv("OUTFIT_AUG",     str(ROOT / "data/processed/polyvore/manifests_disjoint/items_test_aug.csv")))
IMGROOT = Path(os.getenv("OUTFIT_IMAGES",  str(ROOT / "data/processed/polyvore/images")))
IMGROOT.mkdir(parents=True, exist_ok=True)

# ---------------- app ----------------
app = FastAPI(title="Outfit Recommender API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# serve catalog images for frontend preview
app.mount("/catalog", StaticFiles(directory=str(IMGROOT), html=False), name="catalog")

# load recommend engine once on startup
try:
    engine = RecommendEngine(
        ckpt_path=CKPT,
        catalog_csv=CATCSV,
        images_root=IMGROOT,
        items_aug_csv=AUGCSV,
        img_size=224,
        batch_size=256,
        num_workers=0,  # set >0 on Linux for speed if you like
    )
    startup_error = ""
except Exception as e:
    engine = None
    startup_error = str(e)

@app.get("/health")
def health():
    ok = (engine is not None) and (startup_error == "")
    return {"ok": ok, "error": startup_error}

# ---------------- /recommend : catalog flow ----------------
@app.post("/recommend")
async def recommend(
    image: UploadFile = File(...),
    allow_types: str = Query("tops,bottoms,outerwear,footwear"),
    per_bucket: int = Query(5, ge=1, le=20),
    topk: int = Query(20, ge=1, le=100),
    color_weight: float = Query(0.1, ge=0.0, le=1.0),
    anchor_color_mode: str = Query("auto", pattern="^(auto|hsv|km)$"),
):
    if engine is None:
        raise HTTPException(status_code=500, detail=f"Engine not ready: {startup_error}")

    try:
        pil = Image.open(image.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    anchor_color, df = engine.recommend(
        pil,
        allow_types=allow_types,
        per_bucket=per_bucket,
        topk=topk,
        color_weight=color_weight,
        anchor_color_mode=anchor_color_mode,
    )

    # preview_url is relative; frontend will prepend http://127.0.0.1:8000
    def preview_url(rel: str) -> str:
        rel = str(rel).replace("\\", "/")
        return f"/catalog/{rel}"

    items = []
    for _, r in df.iterrows():
        items.append({
            "item_id": r["item_id"],
            "image_path": r["image_path"],
            "preview_url": preview_url(r["image_path"]),
            "category": r["category"],
            "bucket": r["bucket"],
            "title": r["title"],
            "score": float(r["score"]),
        })

    return {
        "anchor_color": anchor_color,
        "allow_types": allow_types,
        "topk": topk,
        "per_bucket": per_bucket,
        "items": items,
    }

# ---------------- wardrobe helpers ----------------
@torch.no_grad()
def _encode_files(files: list[UploadFile], img_size: int, device: torch.device):
    from src.service.recommender import load_image_preproc
    xs, names = [], []
    for f in files:
        try:
            im = Image.open(f.file).convert("RGB")
        except Exception:
            continue
        xs.append(load_image_preproc(im, img_size))
        names.append(getattr(f, "filename", ""))
    if not xs:
        return torch.empty((0,)), []
    xb = torch.stack(xs, dim=0).to(device)
    with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
        feats = engine.model.encoder(xb).detach()
    return feats, names

def _pair_logits(fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
    # fa: [A,D], fb: [B,D] -> logits [A,B] using head on concatenated features
    A, D = fa.shape
    B = fb.shape[0]
    out = []
    # chunk across A to avoid huge cat
    step = max(1, 8192 // max(1, D * 2))
    for i in range(0, A, step):
        f1 = fa[i:i + step]      # [c,D]
        c = f1.shape[0]
        cat = torch.cat(
            [f1.repeat_interleave(B, 0), fb.repeat(c, 1)],
            dim=1
        )  # [c*B, 2D]
        with torch.amp.autocast('cuda', enabled=(engine.device.type == "cuda")):
            logits = engine.model.head(cat).squeeze(1)  # [c*B]
        out.append(logits.view(c, B).detach())
    return torch.cat(out, dim=0)

# ---------------- /wardrobe/recommend : user wardrobe combos ----------------
@app.post("/wardrobe/recommend")
async def wardrobe_recommend(
    tops:      Annotated[list[UploadFile] | None, File()] = None,
    bottoms:   Annotated[list[UploadFile] | None, File()] = None,
    outerwear: Annotated[list[UploadFile] | None, File()] = None,
    footwear:  Annotated[list[UploadFile] | None, File()] = None,
    topk:      Annotated[int, Form()] = 20,
):
    if engine is None:
        raise HTTPException(status_code=500, detail=f"Engine not ready: {startup_error}")

    # normalize None -> []
    tops      = tops or []
    bottoms   = bottoms or []
    outerwear = outerwear or []
    footwear  = footwear or []

    # guard rails (smaller cap is MUCH faster on CPU)
    CAP = int(os.getenv("WARDROBE_CAP", "16"))
    tops      = tops[:CAP]
    bottoms   = bottoms[:CAP]
    outerwear = outerwear[:CAP]
    footwear  = footwear[:CAP]

    with torch.no_grad():
        
        present = [(n, lst) for n, lst in (("tops", tops), ("bottoms", bottoms), ("outerwear", outerwear), ("footwear", footwear)) if len(lst) > 0]
        if len(present) < 2:
            raise HTTPException(status_code=400, detail="Add at least two categories (e.g., tops and bottoms).")

        # encode per bucket
        fT, nT = _encode_files(tops, engine.img_size, engine.device)
        fB, nB = _encode_files(bottoms, engine.img_size, engine.device)
        fO, nO = _encode_files(outerwear, engine.img_size, engine.device) if len(outerwear) else (torch.empty((0,)), [])
        fF, nF = _encode_files(footwear, engine.img_size, engine.device)  if len(footwear)  else (torch.empty((0,)), [])

        # pair tables
        pairs: Dict[Tuple[str, str], torch.Tensor] = {}
        if len(fT) and len(fB): pairs[("tops", "bottoms")]   = _pair_logits(fT, fB)
        if len(fT) and len(fO): pairs[("tops", "outerwear")] = _pair_logits(fT, fO)
        if len(fT) and len(fF): pairs[("tops", "footwear")]  = _pair_logits(fT, fF)
        if len(fB) and len(fO): pairs[("bottoms", "outerwear")] = _pair_logits(fB, fO)
        if len(fB) and len(fF): pairs[("bottoms", "footwear")]  = _pair_logits(fB, fF)
        if len(fO) and len(fF): pairs[("outerwear", "footwear")] = _pair_logits(fO, fF)

        outfits = []

        def add_two(catA: str, catB: str, namesA: List[str], namesB: List[str], mat: torch.Tensor):
            A, B = mat.shape
            scores = torch.sigmoid(mat)
            vals, idxs = torch.topk(scores.flatten(), k=min(topk * 5, A * B))
            for v, flat in zip(vals.tolist(), idxs.tolist()):
                i = flat // B
                j = flat % B
                outfits.append({
                    "score": float(v),
                    "parts": [
                        {"slot": catA, "idx": int(i), "name": namesA[i]},
                        {"slot": catB, "idx": int(j), "name": namesB[j]},
                    ],
                })

        # simple 2-bucket case
        if len(fT) and len(fB) and not len(fO) and not len(fF):
            add_two("tops", "bottoms", nT, nB, pairs[("tops", "bottoms")])
        else:
            # seed pair: prefer (tops,bottoms), else the largest table available
            if ("tops", "bottoms") in pairs:
                a, b = "tops", "bottoms"
                nA, nB = nT, nB
                matAB = pairs[(a, b)]
            else:
                if not pairs:
                    raise HTTPException(status_code=400, detail="Not enough categories with valid images.")
                (a, b), matAB = max(pairs.items(), key=lambda kv: kv[1].numel())
                nA = {"tops": nT, "bottoms": nB, "outerwear": nO, "footwear": nF}[a]
                nB = {"tops": nT, "bottoms": nB, "outerwear": nO, "footwear": nF}[b]

            A, B = matAB.shape
            beam_k = min(max(topk * 10, 100), A * B)
            vals, idxs = torch.topk(torch.sigmoid(matAB).flatten(), k=beam_k)
            beams = [(float(vals[i]), int(idxs[i] // B), int(idxs[i] % B)) for i in range(beam_k)]

            def try_extend(beams_in, cat: str, namesC: List[str]):
                featsC = {"tops": fT, "bottoms": fB, "outerwear": fO, "footwear": fF}[cat]
                if len(featsC) == 0:
                    return beams_in
                out = []
                keyA = (a, cat) if (a, cat) in pairs else (cat, a) if (cat, a) in pairs else None
                keyB = (b, cat) if (b, cat) in pairs else (cat, b) if (cat, b) in pairs else None
                matAC = pairs.get(keyA) if keyA else None
                matBC = pairs.get(keyB) if keyB else None
                for base_score, ia, ib in beams_in:
                    if matAC is None and matBC is None:
                        continue
                    # per-candidate scores for C
                    if matAC is not None:
                        s_ac = torch.sigmoid(matAC[ia] if keyA == (a, cat) else matAC[:, ia])
                    else:
                        s_ac = None
                    if matBC is not None:
                        s_bc = torch.sigmoid(matBC[ib] if keyB == (b, cat) else matBC[:, ib])
                    else:
                        s_bc = None
                    if s_ac is not None and s_bc is not None:
                        s = (s_ac + s_bc) / 2.0
                    else:
                        s = s_ac if s_ac is not None else s_bc
                    topv, topi = torch.topk(s, k=min(50, s.shape[0]))
                    for v, ic in zip(topv.tolist(), topi.tolist()):
                        out.append(((base_score + float(v)) / 2.0, ia, ib, int(ic)))
                out.sort(key=lambda x: x[0], reverse=True)
                return out[: max(topk * 20, 100)]

            beams2 = beams
            remaining = [c for c in ("outerwear", "footwear") if c not in (a, b)]
            for cat in remaining:
                namesC = {"tops": nT, "bottoms": nB, "outerwear": nO, "footwear": nF}[cat]
                beams2 = try_extend(beams2, cat, namesC)

            # finalize outfits from beams2
            for tpl in beams2[: topk * 5]:
                base, ia, ib, *rest = tpl
                parts = [
                    {"slot": a, "idx": ia, "name": nA[ia]},
                    {"slot": b, "idx": ib, "name": nB[ib]},
                ]
                if len(rest) >= 1:
                    ic = rest[0]
                    slotC = remaining[0] if len(remaining) >= 1 else None
                    if slotC:
                        namesC = {"outerwear": nO, "footwear": nF}[slotC]
                        parts.append({"slot": slotC, "idx": ic, "name": namesC[ic]})
                outfits.append({"score": float(base), "parts": parts})

        outfits.sort(key=lambda x: x["score"], reverse=True)
        outfits = outfits[: topk]
    return {"topk": topk, "items": outfits}
