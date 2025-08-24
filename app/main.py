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
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project Setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service.recommender import RecommendEngine

# Configuration
class Config:
    CKPT = Path(os.getenv("OUTFIT_CKPT", str(ROOT / "models" / "compatibility_best.pt")))
    CATCSV = Path(os.getenv("OUTFIT_CATALOG", str(ROOT / "data/processed/polyvore/manifests_disjoint/items_test.csv")))
    AUGCSV = Path(os.getenv("OUTFIT_AUG", str(ROOT / "data/processed/polyvore/manifests_disjoint/items_test_aug.csv")))
    IMGROOT = Path(os.getenv("OUTFIT_IMAGES", str(ROOT / "data/processed/polyvore/images")))
    ITEM_ONNX = Path(os.getenv("OUTFIT_ITEM_ONNX", str(ROOT / "models" / "item_classifier" / "item_classifier.onnx")))
    ITEM_META = Path(os.getenv("OUTFIT_ITEM_META", str(ROOT / "models" / "item_classifier" / "item_classifier_meta.json")))
    FEATS_CACHE = Path(os.getenv("OUTFIT_FEATS_CACHE", str(ROOT / "data" / "cache" / "catalog_feats.pt")))
    
    # Performance settings
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
    DEVICE = os.getenv("DEVICE", "auto")
    WARDROBE_CAP = int(os.getenv("WARDROBE_CAP", "8"))
    
    # Timeout settings
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))  # Increased for debugging
    
config = Config()

# Response Models
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

# Global Engine
engine: Optional[RecommendEngine] = None
startup_error: str = ""

# Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global engine, startup_error
    try:
        logger.info("Starting outfit recommendation system...")
        logger.info(f"Using catalog: {config.CATCSV}")
        logger.info(f"Using images: {config.IMGROOT}")
        logger.info(f"Using cache: {config.FEATS_CACHE}")
        
        # Determine device
        device = config.DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        
        logger.info(f"Using device: {device}")
        
        # Create engine with all parameters
        engine = RecommendEngine(
            ckpt_path=config.CKPT,
            catalog_csv=config.CATCSV,
            images_root=config.IMGROOT,
            items_aug_csv=config.AUGCSV if config.AUGCSV.exists() else None,
            img_size=224,
            batch_size=config.MAX_BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            device=device,
            item_cls_onnx=config.ITEM_ONNX if config.ITEM_ONNX.exists() else None,
            item_cls_meta=config.ITEM_META if config.ITEM_META.exists() else None,
            feats_cache=config.FEATS_CACHE if config.FEATS_CACHE.exists() else None,
        )
        
        # Warmup
        logger.info("Warming up model...")
        await warmup_model()
        
        logger.info("Recommendation engine loaded successfully")
        logger.info(f"Loaded {len(engine.items)} catalog items")
        logger.info(f"Feature cache: {'loaded' if engine.catalog_feats is not None else 'not available'}")
        startup_error = ""
        
    except Exception as e:
        logger.error(f"Failed to load recommendation engine: {e}")
        import traceback
        traceback.print_exc()
        engine = None
        startup_error = str(e)
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if engine:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# App Initialization
app = FastAPI(
    title="Outfit Recommender API",
    version="0.3.0",
    description="AI-powered outfit recommendation system with improved filtering and color detection",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
config.IMGROOT.mkdir(parents=True, exist_ok=True)
app.mount("/catalog", StaticFiles(directory=str(config.IMGROOT), html=False), name="catalog")

# Utility Functions
async def warmup_model():
    """Warmup the model with a dummy image"""
    if engine is None:
        return
    
    try:
        dummy_img = Image.new('RGB', (224, 224), color='white')
        
        await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: engine.recommend(
                dummy_img,
                allow_types="tops,bottoms",
                per_bucket=1,
                topk=1,
                color_weight=0.0,
                filter_same_bucket=False,
                anchor_color_mode="hsv",
            )
        )
        logger.info("Model warmup completed")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")

def validate_image(image: UploadFile) -> Image.Image:
    """Validate and process uploaded image"""
    try:
        # Check file size (10MB limit)
        if hasattr(image, 'size') and image.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
        
        # Try to open and validate
        pil_img = Image.open(image.file).convert("RGB")
        
        # Check dimensions
        if pil_img.size[0] < 50 or pil_img.size[1] < 50:
            raise HTTPException(status_code=400, detail="Image too small (minimum 50x50 pixels)")
        
        return pil_img
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def create_preview_url(rel_path: str) -> str:
    """Create preview URL for images"""
    rel_path = str(rel_path).replace("\\", "/")
    return f"/catalog/{rel_path}"

# API Endpoints

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if engine is not None else "unhealthy",
        "engine_loaded": engine is not None,
        "error": startup_error if startup_error else None,
        "device": str(engine.device) if engine else None,
        "catalog_size": len(engine.items) if engine and hasattr(engine, 'items') else 0,
        "feature_cache": engine.catalog_feats is not None if engine else False,
        "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    }

@app.post("/recommend", response_model=CatalogResponse)
async def recommend(
    image: UploadFile = File(...),
    allow_types: str = Query("tops,bottoms,outerwear,footwear", regex=r"^[a-z,]+$"),
    per_bucket: int = Query(5, ge=1, le=10),
    topk: int = Query(20, ge=1, le=50),
    color_weight: float = Query(0.1, ge=0.0, le=1.0),
    anchor_color_mode: str = Query("auto", regex=r"^(auto|hsv|km)$"),
    filter_same_bucket: bool = Query(True),
):
    """Get catalog recommendations for a user image"""
    
    if engine is None:
        raise HTTPException(status_code=503, detail=f"Engine not ready: {startup_error}")
    
    # Validate image
    pil_img = validate_image(image)
    logger.info(f"Processing recommendation request: allow_types={allow_types}, filter_same_bucket={filter_same_bucket}")
    
    try:
        # Add debug logging
        logger.info(f"Request params: color_weight={color_weight}, anchor_color_mode={anchor_color_mode}")
        
        # Run recommendation in thread pool
        start_time = asyncio.get_event_loop().time()
        anchor_color, df = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: engine.recommend(
                    pil_img,
                    allow_types=allow_types,
                    per_bucket=per_bucket,
                    topk=topk,
                    color_weight=color_weight,  # Make sure this is passed
                    anchor_color_mode=anchor_color_mode,
                    filter_same_bucket=filter_same_bucket,
                )
            ),
            timeout=config.REQUEST_TIMEOUT
        )
        
        # Add debug logging for results
        if len(df) > 0:
            logger.info(f"Score range: {df['score'].min():.3f} - {df['score'].max():.3f}")
        
        processing_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Recommendation completed in {processing_time:.2f}s, found {len(df)} items")
        
        # Convert to response format
        items = []
        for _, row in df.iterrows():
            try:
                items.append(RecommendationItem(
                    item_id=str(row["item_id"]),
                    image_path=str(row["image_path"]),
                    preview_url=create_preview_url(row["image_path"]),
                    category=str(row["category"]),
                    bucket=str(row["bucket"]),
                    title=str(row["title"]),
                    score=float(row["score"]),
                ))
            except Exception as e:
                logger.warning(f"Skipping item due to conversion error: {e}")
                continue
        
        return CatalogResponse(
            anchor_color=anchor_color,
            allow_types=allow_types,
            topk=topk,
            per_bucket=per_bucket,
            items=items
        )
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout - please try again")
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/wardrobe/recommend", response_model=WardrobeResponse)
async def wardrobe_recommend(
    tops: Annotated[Optional[List[UploadFile]], File()] = None,
    bottoms: Annotated[Optional[List[UploadFile]], File()] = None,
    outerwear: Annotated[Optional[List[UploadFile]], File()] = None,
    footwear: Annotated[Optional[List[UploadFile]], File()] = None,
    topk: Annotated[int, Form()] = 10,
):
    """Get wardrobe outfit combinations"""
    
    if engine is None:
        raise HTTPException(status_code=503, detail=f"Engine not ready: {startup_error}")
    
    # Normalize None -> []
    tops = tops or []
    bottoms = bottoms or []
    outerwear = outerwear or []
    footwear = footwear or []
    # Apply caps to prevent resource exhaustion
    tops = tops[:config.WARDROBE_CAP]
    bottoms = bottoms[:config.WARDROBE_CAP]
    outerwear = outerwear[:config.WARDROBE_CAP]
    footwear = footwear[:config.WARDROBE_CAP]
    
    # Validate minimum requirements
    total_items = len(tops) + len(bottoms) + len(outerwear) + len(footwear)
    if total_items < 2:
        raise HTTPException(
            status_code=400, 
            detail="At least 2 clothing items required across categories"
        )
    
    # Check if we have at least 2 categories
    categories_with_items = sum([len(tops) > 0, len(bottoms) > 0, len(outerwear) > 0, len(footwear) > 0])
    if categories_with_items < 2:
        raise HTTPException(
            status_code=400,
            detail="Add at least two categories (e.g., tops and bottoms)"
        )
    
    logger.info(f"Processing wardrobe recommendation: {len(tops)} tops, {len(bottoms)} bottoms, {len(outerwear)} outerwear, {len(footwear)} footwear")
    
    try:
        # Process in thread pool
        start_time = asyncio.get_event_loop().time()
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda: engine.wardrobe_recommend(
                    tops=tops,
                    bottoms=bottoms,
                    outerwear=outerwear,
                    footwear=footwear,
                    topk=min(topk, 20)  # Cap topk
                )
            ),
            timeout=config.REQUEST_TIMEOUT * 2  # Longer timeout for wardrobe
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Wardrobe recommendation completed in {processing_time:.2f}s, generated {len(result['items'])} outfits")
        
        return WardrobeResponse(
            topk=topk,
            items=[
                OutfitCombo(
                    score=combo["score"],
                    parts=[
                        OutfitPart(
                            slot=part["slot"],
                            idx=part["idx"],
                            name=part["name"]
                        )
                        for part in combo["parts"]
                    ]
                )
                for combo in result["items"]
            ]
        )
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout - wardrobe analysis taking too long")
    except Exception as e:
        logger.error(f"Wardrobe recommendation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Wardrobe analysis failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system status information"""
    if engine is None:
        return {"status": "engine_not_loaded", "error": startup_error}
    
    status_info = {
        "status": "ready",
        "device": str(engine.device),
        "model_loaded": True,
        "head_loaded": bool(getattr(engine, "_head_ok", False)),
        "catalog_size": len(engine.items) if hasattr(engine, 'items') else 0,
        "batch_size": engine.batch_size if hasattr(engine, 'batch_size') else 0,
        "feature_cache_loaded": engine.catalog_feats is not None,
        "color_lookup_size": len(engine.color_lookup) if hasattr(engine, 'color_lookup') else 0,
    }
    
    if torch.cuda.is_available():
        status_info.update({
            "cuda_available": True,
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "gpu_memory_reserved": torch.cuda.memory_reserved(),
        })
    
    return status_info

@app.get("/debug/catalog-stats")
async def get_catalog_stats():
    """Get detailed catalog statistics for debugging"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    bucket_stats = engine.items['bucket'].value_counts().to_dict()
    category_stats = engine.items['category'].value_counts().head(20).to_dict()
    
    return {
        "total_items": len(engine.items),
        "bucket_distribution": bucket_stats,
        "top_categories": category_stats,
        "sample_items": engine.items.head(5).to_dict('records'),
        "feature_cache": engine.catalog_feats is not None,
        "color_data": len(engine.color_lookup) if hasattr(engine, 'color_lookup') else 0,
    }

# Error Handlers
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
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500},
    )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,
    )