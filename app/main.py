import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Annotated
from contextlib import asynccontextmanager

import torch
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Imports – align with recommender.py (AdvancedRecommendationEngine)
# ----------------------------------------------------------------------------
try:
    # If recommender.py is in the same module path
    from recommender import (
        AdvancedRecommendationEngine,
        PerformanceConfig,
        get_optimal_config,
    )
except Exception:
    # Fallback to package-style import if your repo uses src/ layout
    from src.service.recommender import (
        AdvancedRecommendationEngine,
        PerformanceConfig,
        get_optimal_config,
    )

# ----------------------------------------------------------------------------
# Project Setup
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ----------------------------------------------------------------------------
# Configuration – only pass what AdvancedRecommendationEngine expects
# ----------------------------------------------------------------------------
class Config:
    CKPT = Path(os.getenv("OUTFIT_CKPT", str(ROOT / "models" / "compatibility_best.pt")))
    CATCSV = Path(os.getenv("OUTFIT_CATALOG", str(ROOT / "data/processed/polyvore/manifests_disjoint/items_test.csv")))
    AUGCSV = Path(os.getenv("OUTFIT_AUG", str(ROOT / "data/processed/polyvore/manifests_disjoint/items_test_aug.csv")))
    IMGROOT = Path(os.getenv("OUTFIT_IMAGES", str(ROOT / "data/processed/polyvore/images")))
    ITEM_ONNX = Path(os.getenv("OUTFIT_ITEM_ONNX", str(ROOT / "models" / "item_classifier" / "item_classifier.onnx")))
    ITEM_META = Path(os.getenv("OUTFIT_ITEM_META", str(ROOT / "models" / "item_classifier" / "item_classifier_meta.json")))

    DEVICE = os.getenv("DEVICE", "auto")
    WARDROBE_CAP = int(os.getenv("WARDROBE_CAP", "8"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

config = Config()

# ----------------------------------------------------------------------------
# Response Models (stable contract for the frontend)
# ----------------------------------------------------------------------------
class RecommendationItem(BaseModel):
    item_id: str
    image_path: str
    preview_url: str
    category: str
    bucket: str
    title: str
    score: float

class CatalogResponse(BaseModel):
    anchor_color: str
    allow_types: str
    topk: int
    per_bucket: int
    items: List[RecommendationItem]

class OutfitPart(BaseModel):
    slot: str
    idx: int
    name: str

class OutfitCombo(BaseModel):
    score: float
    parts: List[OutfitPart]

class WardrobeResponse(BaseModel):
    topk: int
    items: List[OutfitCombo]

# ----------------------------------------------------------------------------
# Global Engine
# ----------------------------------------------------------------------------
engine: Optional[AdvancedRecommendationEngine] = None
startup_error: str = ""

# ----------------------------------------------------------------------------
# Lifespan – instantiate AdvancedRecommendationEngine and warm it up
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, startup_error
    try:
        logger.info("Starting outfit recommendation system (aligned with AdvancedRecommendationEngine)...")
        logger.info(f"Using catalog: {config.CATCSV}")
        logger.info(f"Using images: {config.IMGROOT}")

        # Device
        device = config.DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        logger.info(f"Using device: {device}")

        # Perf config (let recommender auto-tune by default)
        perf_cfg: PerformanceConfig = get_optimal_config()

        # Engine – NOTE: AdvancedRecommendationEngine does NOT take feats_cache/batch_size args
        engine = AdvancedRecommendationEngine(
            ckpt_path=config.CKPT,
            catalog_csv=config.CATCSV,
            images_root=config.IMGROOT,
            items_aug_csv=config.AUGCSV if config.AUGCSV.exists() else None,
            img_size=224,
            device=device,
            item_cls_onnx=config.ITEM_ONNX if config.ITEM_ONNX.exists() else None,
            item_cls_meta=config.ITEM_META if config.ITEM_META.exists() else None,
            config=perf_cfg,
        )

        # Warmup once with a tiny request hitting the encoder & path
        logger.info("Warming up model…")
        await warmup_model()
        logger.info("Engine ready.")
        startup_error = ""

    except Exception as e:
        logger.exception("Failed to load recommendation engine")
        engine = None
        startup_error = str(e)

    yield

    # Shutdown cleanup
    logger.info("Shutting down…")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ----------------------------------------------------------------------------
# App Init + CORS + Static
# ----------------------------------------------------------------------------
app = FastAPI(
    title="Outfit Recommender API (Aligned)",
    version="0.4.0",
    description="API aligned to AdvancedRecommendationEngine with wardrobe pairing",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config.IMGROOT.mkdir(parents=True, exist_ok=True)
app.mount("/catalog", StaticFiles(directory=str(config.IMGROOT), html=False), name="catalog")

# ----------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------
async def warmup_model():
    if engine is None:
        return
    try:
        dummy = Image.new('RGB', (224, 224), color='white')
        # Use the new recommend_advanced entrypoint
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: engine.recommend_advanced(
                dummy,
                allow_types="tops,bottoms",
                per_bucket=1,
                topk=1,
                color_weight=0.0,
                style_weight=0.0,
                diversity_weight=0.0,
                anchor_color_mode="hsv",
                filter_same_bucket=False,
            ),
        )
        logger.info("Model warmup completed")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")


def validate_image(image: UploadFile) -> Image.Image:
    try:
        pil_img = Image.open(image.file).convert("RGB")
        if pil_img.size[0] < 50 or pil_img.size[1] < 50:
            raise HTTPException(status_code=400, detail="Image too small (min 50x50)")
        return pil_img
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


def create_preview_url(rel_path: str) -> str:
    rel_path = str(rel_path).replace("\\", "/")
    return f"/catalog/{rel_path}"

# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy" if engine is not None else "unhealthy",
        "engine_loaded": engine is not None,
        "error": startup_error or None,
        "device": str(engine.device) if engine else None,
        "catalog_size": int(len(engine.items)) if engine and hasattr(engine, 'items') else 0,
        # Advanced engine exposes get_performance_stats()
        "perf": engine.get_performance_stats() if engine else {},
    }


@app.post("/recommend", response_model=CatalogResponse)
async def recommend(
    image: UploadFile = File(...),
    allow_types: str = Query("tops,bottoms,outerwear,footwear", regex=r"^[a-z,]+$"),
    per_bucket: int = Query(5, ge=1, le=10),
    topk: int = Query(20, ge=1, le=50),
    color_weight: float = Query(0.15, ge=0.0, le=1.0),
    style_weight: float = Query(0.10, ge=0.0, le=1.0),
    diversity_weight: float = Query(0.05, ge=0.0, le=1.0),
    anchor_color_mode: str = Query("auto", regex=r"^(advanced|kmeans|hsv|auto|km)$"),
    filter_same_bucket: bool = Query(True),
):
    if engine is None:
        raise HTTPException(status_code=503, detail=f"Engine not ready: {startup_error}")

    pil_img = validate_image(image)
    mode = {"auto": "advanced", "km": "kmeans"}.get(anchor_color_mode, anchor_color_mode)
    
    try:
        start = asyncio.get_event_loop().time()
        anchor_color, df, analytics = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: engine.recommend_advanced(
                pil_img,
                allow_types=allow_types,
                per_bucket=per_bucket,
                topk=topk,
                color_weight=color_weight,
                style_weight=style_weight,
                diversity_weight=diversity_weight,
                anchor_color_mode=mode,
                filter_same_bucket=filter_same_bucket,
            ),
        )
        elapsed = asyncio.get_event_loop().time() - start
        logger.info(f"/recommend finished in {elapsed:.2f}s, {len(df)} items")

        items: List[RecommendationItem] = []
        for _, row in df.iterrows():
            items.append(
                RecommendationItem(
                    item_id=str(row.get("item_id", "")),
                    image_path=str(row.get("image_path", "")),
                    preview_url=create_preview_url(row.get("image_path", "")),
                    category=str(row.get("category", "")),
                    bucket=str(row.get("bucket", "")),
                    title=str(row.get("title", "")),
                    score=float(row.get("score", 0.0)),
                )
            )

        return CatalogResponse(
            anchor_color=anchor_color,
            allow_types=allow_types,
            topk=topk,
            per_bucket=per_bucket,
            items=items,
        )

    except Exception as e:
        logger.exception("Recommendation failed")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")


@app.post("/wardrobe/recommend", response_model=WardrobeResponse)
async def wardrobe_recommend(
    tops: Annotated[Optional[List[UploadFile]], File()] = None,
    bottoms: Annotated[Optional[List[UploadFile]], File()] = None,
    outerwear: Annotated[Optional[List[UploadFile]], File()] = None,
    footwear: Annotated[Optional[List[UploadFile]], File()] = None,
    topk: Annotated[int, Form()] = 10,
):
    if engine is None:
        raise HTTPException(status_code=503, detail=f"Engine not ready: {startup_error}")

    # Normalize & cap per bucket to protect resources
    tops = (tops or [])[:config.WARDROBE_CAP]
    bottoms = (bottoms or [])[:config.WARDROBE_CAP]
    outerwear = (outerwear or [])[:config.WARDROBE_CAP]
    footwear = (footwear or [])[:config.WARDROBE_CAP]

    cat_lists: Dict[str, List[UploadFile]] = {
        "tops": tops,
        "bottoms": bottoms,
        "outerwear": outerwear,
        "footwear": footwear,
    }

    n_items = sum(len(v) for v in cat_lists.values())
    if n_items < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 clothing items across categories")
    if sum(1 for v in cat_lists.values() if v) < 2:
        raise HTTPException(status_code=400, detail="Add at least two categories (e.g., tops and bottoms)")

    def _encode_files(files: List[UploadFile]):
        tensors: List[torch.Tensor] = []
        names: List[str] = []
        for f in files:
            try:
                img = Image.open(f.file).convert("RGB")
                t = engine.preprocessor.preprocess_image(img)
                tensors.append(t)
                names.append(getattr(f, "filename", "item"))
            except Exception:
                continue
        if not tensors:
            return torch.empty(0, 3, engine.img_size, engine.img_size), []
        batch = torch.stack(tensors, dim=0).to(engine.device, non_blocking=True)
        with torch.amp.autocast(device_type=engine.device.type, enabled=engine.config.mixed_precision):
            feats = engine.model.encoder(batch)
        # L2 normalize for cosine
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats, names

    # Encode per bucket
    feats_map: Dict[str, torch.Tensor] = {}
    names_map: Dict[str, List[str]] = {}
    for bucket, files in cat_lists.items():
        feats, names = await asyncio.get_event_loop().run_in_executor(None, _encode_files, files)
        feats_map[bucket] = feats
        names_map[bucket] = names

    # Build pairwise combos across present categories using cosine similarity
    combos: List[Tuple[float, List[Dict[str, object]]]] = []
    present = [b for b, t in feats_map.items() if t.numel() > 0]

    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            Fa, Fb = feats_map[a], feats_map[b]
            if Fa.numel() == 0 or Fb.numel() == 0:
                continue
            # cosine matrix
            sim = Fa @ Fb.T  # already normalized, so dot == cosine
            sim_np = sim.detach().float().cpu().numpy()
            for ia in range(sim_np.shape[0]):
                for ib in range(sim_np.shape[1]):
                    score = float((sim_np[ia, ib] + 1.0) * 0.5)  # map [-1,1] -> [0,1]
                    combos.append(
                        (
                            score,
                            [
                                {"slot": a, "idx": ia, "name": names_map[a][ia] if ia < len(names_map[a]) else f"{a}-{ia}"},
                                {"slot": b, "idx": ib, "name": names_map[b][ib] if ib < len(names_map[b]) else f"{b}-{ib}"},
                            ],
                        )
                    )

    combos.sort(key=lambda x: x[0], reverse=True)
    top = combos[: min(max(topk, 1), 20)]

    return WardrobeResponse(
        topk=topk,
        items=[
            OutfitCombo(
                score=s,
                parts=[OutfitPart(**p) for p in parts],
            )
            for (s, parts) in top
        ],
    )


@app.get("/status")
async def get_status():
    if engine is None:
        return {"status": "engine_not_loaded", "error": startup_error}

    info = engine.get_performance_stats()
    return {
        "status": "ready",
        "device": str(engine.device),
        "catalog_size": len(engine.items) if hasattr(engine, 'items') else 0,
        "color_lookup_size": len(getattr(engine, 'color_lookup', {})),
        "perf": info,
    }


@app.get("/debug/catalog-stats")
async def get_catalog_stats():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    bucket_stats = engine.items['bucket'].value_counts().to_dict()
    category_stats = engine.items['category'].value_counts().head(20).to_dict()

    return {
        "total_items": len(engine.items),
        "bucket_distribution": bucket_stats,
        "top_categories": category_stats,
        "sample_items": engine.items.head(5).to_dict('records'),
        "color_data": len(getattr(engine, 'color_lookup', {})),
    }

# ----------------------------------------------------------------------------
# Error Handlers
# ----------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500},
    )

# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_aligned:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,
    )
