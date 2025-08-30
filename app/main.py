import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Annotated, Any
from contextlib import asynccontextmanager
import numpy as np

from datetime import datetime
import calendar

import torch
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import time
import json
import httpx
from functools import lru_cache

# Try package-aware imports first, then fallback to top-level modules.
try:
    from app.fashion_rules import FashionCompatibilityEngine, AdvancedFormalityClassifier  # files live in app/
    from app.enhanced_wardrobe_integration import EnhancedWardrobeRecommender
except Exception:
    try:
        from fashion_rules import FashionCompatibilityEngine, AdvancedFormalityClassifier  # files at repo root
        from enhanced_wardrobe_integration import EnhancedWardrobeRecommender
    except Exception:
        FashionCompatibilityEngine = None  # type: ignore
        AdvancedFormalityClassifier = None  # type: ignore
        EnhancedWardrobeRecommender = None  # type: ignore


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Imports â€“ align with recommender.py (AdvancedRecommendationEngine)
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
    
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# ----------------------------------------------------------------------------
# Configuration â€“ only pass what AdvancedRecommendationEngine expects
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

class EnhancedConfig(Config):
    """Enhanced configuration with fashion rule parameters"""
    ENABLE_FASHION_RULES = os.getenv("ENABLE_FASHION_RULES", "true").lower() == "true"
    FORMALITY_WEIGHT = float(os.getenv("FORMALITY_WEIGHT", "0.4"))
    STRICT_RULES_MODE = os.getenv("STRICT_RULES_MODE", "true").lower() == "true"
    
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
    # Optional, UI ignores if absent
    explain: Optional[Dict[str, Any]] = None

class WardrobeResponse(BaseModel):
    topk: int
    items: List[OutfitCombo]

# ----------------------------------------------------------------------------
# Global Engine
# ----------------------------------------------------------------------------
engine: Optional[AdvancedRecommendationEngine] = None
startup_error: str = ""

enhanced_config = EnhancedConfig()
fashion_engine: Optional[Any] = None
enhanced_recommender: Optional[Any] = None

# ----------------------------------------------------------------------------
# Lifespan â€“ instantiate AdvancedRecommendationEngine and warm it up
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, startup_error, fashion_engine, enhanced_recommender
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

        # Engine â€“ NOTE: AdvancedRecommendationEngine does NOT take feats_cache/batch_size args
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
        
        # Initialize fashion rule components
        if enhanced_config.ENABLE_FASHION_RULES and (EnhancedWardrobeRecommender is not None) and (FashionCompatibilityEngine is not None):
            try:
                fashion_engine = FashionCompatibilityEngine()
                enhanced_recommender = EnhancedWardrobeRecommender(engine, enhanced_config)
                logger.info("Fashion rule engine initialized successfully")
            except Exception as e:
                logger.warning(f"Fashion rule engine failed to initialize: {e}")
                fashion_engine = None
                enhanced_recommender = None
        else:
            logger.info("Fashion rules disabled or modules not found; continuing without them")


        # Warmup once with a tiny request hitting the encoder & path
        logger.info("Warming up modelâ€¦")
        await warmup_model()
        logger.info("Engine ready.")
        startup_error = ""

    except Exception as e:
        logger.exception("Failed to load recommendation engine")
        engine = None
        startup_error = str(e)

    yield

    # Shutdown cleanup
    logger.info("Shutting downâ€¦")
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
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
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

def _hint_from_metrics(temp_c: float, precip_mm: float, wind_kph: float, code: int) -> str:
    """Basic weather hint determination (fallback version)"""
    if precip_mm >= 0.5 or code in {51,53,55,56,57,61,63,65,66,67,80,81,82,95,96,99}:
        return "rain"
    
    if temp_c <= 5.0:
        return "cold"
    elif temp_c >= 28.0:
        return "hot"
    elif wind_kph > 20:
        return "cold" if temp_c < 18 else "mild"
    
    return "mild"
# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------

@app.get("/weather/analytics")
async def weather_analytics(lat: float, lon: float):
    """Get weather trends and analytics for a location."""
    key = _wx_cache_key(lat, lon)
    
    if key not in _weather_trends or not _weather_trends[key]:
        raise HTTPException(status_code=404, detail="No weather history for this location")
    
    trends = _weather_trends[key]
    recent_data = [data for _, data in trends[-12:]]  # Last 12 data points
    
    if not recent_data:
        raise HTTPException(status_code=404, detail="Insufficient weather data")
    
    temps = [d.get("temp_c", 0) for d in recent_data]
    feels_like = [d.get("feels_like", d.get("temp_c", 0)) for d in recent_data]
    
    analytics = {
        "location": {"lat": lat, "lon": lon},
        "data_points": len(recent_data),
        "temperature": {
            "avg": round(sum(temps) / len(temps), 1),
            "min": round(min(temps), 1),
            "max": round(max(temps), 1),
            "trend": "warming" if temps[-1] > temps[0] else "cooling" if temps[-1] < temps[0] else "stable"
        },
        "comfort": {
            "avg_feels_like": round(sum(feels_like) / len(feels_like), 1),
            "comfort_variance": round(max(feels_like) - min(feels_like), 1)
        },
        "seasonal_context": _get_seasonal_context(lat),
        "current_hint": recent_data[-1].get("hint", "unknown")
    }
    
    return analytics

@app.get("/fashion/analytics")
async def get_fashion_analytics():
    """Get fashion rule engine analytics"""
    
    if fashion_engine is None:
        return {"status": "fashion_rules_disabled"}
    
    return {
        "status": "active",
        "config": {
            "enabled": enhanced_config.ENABLE_FASHION_RULES,
            "formality_weight": enhanced_config.FORMALITY_WEIGHT,
            "strict_mode": enhanced_config.STRICT_RULES_MODE
        },
        "supported_formalities": ["formal", "smart_casual", "casual", "neutral"],
        "rule_engine_version": "1.0"
    }

@app.post("/fashion/classify")
async def classify_fashion_item(
    image: UploadFile = File(...),
    category: str = Form(...),
    description: str = Form("")
):
    """Classify a single fashion item for diagnostic purposes"""
    
    if fashion_engine is None:
        raise HTTPException(status_code=503, detail="Fashion rule engine not available")
    
    try:
        pil_img = validate_image(image)
        item_name = getattr(image, 'filename', category)
        
        # Classify using fashion rules
        classified_item = fashion_engine.classifier.classify_item(
            item_name=item_name,
            category=category,
            description=description
        )
        
        return {
            "classification": {
                "name": classified_item.name,
                "category": classified_item.category,
                "formality": classified_item.formality,
                "style_category": classified_item.style_category,
                "season_tags": classified_item.season_tags,
                "fabric_type": classified_item.fabric_type,
                "color_family": classified_item.color_family
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

# --- Live Weather Proxy (Open-Meteo) ---
@lru_cache(maxsize=256)
def _wx_cache_key(lat: float, lon: float) -> str:
    return f"{round(lat,2)},{round(lon,2)}"

_weather_cache: Dict[str, Tuple[float, dict]] = {}  # TTL cache (10 minutes)
_weather_trends: Dict[str, List[Tuple[float, dict]]] = {}  # Track weather history for trends

async def _fetch_weather(lat: float, lon: float) -> dict:
    key = _wx_cache_key(lat, lon)
    now = time.time()
    
    # Check cache first
    if key in _weather_cache and now - _weather_cache[key][0] < 600:
        return _weather_cache[key][1]

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,precipitation,weather_code,wind_speed_10m,relative_humidity_2m"
        "&hourly=temperature_2m,precipitation_probability&forecast_days=1"
    )
    
    async with httpx.AsyncClient(timeout=8.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        j = r.json() or {}
        
        cur = j.get("current", {})
        hourly = j.get("hourly", {})
        
        temp_c = float(cur.get("temperature_2m", 0.0))
        precip_mm = float(cur.get("precipitation", 0.0))
        wind_kph = float(cur.get("wind_speed_10m", 0.0))
        humidity = float(cur.get("relative_humidity_2m", 0.0))
        code = int(cur.get("weather_code", 0))
        
        # Calculate feels-like temperature
        feels_like = temp_c
        if wind_kph > 5:  # Wind chill effect
            feels_like -= (wind_kph - 5) * 0.2
        if humidity > 70 and temp_c > 20:  # Heat index effect
            feels_like += (humidity - 70) * 0.1
        
        # Analyze upcoming weather trend
        upcoming_temps = hourly.get("temperature_2m", [])[:6]  # Next 6 hours
        upcoming_precip = hourly.get("precipitation_probability", [])[:6]
        
        temp_trend = "stable"
        if len(upcoming_temps) >= 3:
            if upcoming_temps[-1] > upcoming_temps[0] + 3:
                temp_trend = "warming"
            elif upcoming_temps[-1] < upcoming_temps[0] - 3:
                temp_trend = "cooling"
        
        rain_likelihood = max(upcoming_precip) if upcoming_precip else 0
        
        hint = _hint_from_metrics(temp_c, precip_mm, wind_kph, code)
        
        data = {
            "temp_c": temp_c,
            "feels_like": feels_like,
            "precip_mm": precip_mm,
            "wind_kph": wind_kph,
            "humidity": humidity,
            "code": code,
            "hint": hint,
            "temp_trend": temp_trend,
            "rain_likelihood": rain_likelihood,
        }
        
        # Store in both cache and trends
        _weather_cache[key] = (now, data)
        
        # --- Track recent weather points for analytics ---
        _weather_trends.setdefault(key, []).append((now, data))
        # keep the last ~72 points (~12 hours at 10-min TTL hits)
        if len(_weather_trends[key]) > 72:
            _weather_trends[key] = _weather_trends[key][-72:]
        
        if key not in _weather_trends:
            _weather_trends[key] = []
        _weather_trends[key].append((now, data))
        # Keep only last 24 hours of data
        _weather_trends[key] = [(t, d) for t, d in _weather_trends[key] if now - t < 86400]
        
        return data
    
@app.get("/weather")
async def weather_proxy(lat: float, lon: float):
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise HTTPException(status_code=400, detail="Invalid coordinates")
    try:
        data = await _fetch_weather(lat, lon)
        return data
    except Exception as e:
        logger.warning(f"Weather proxy failed: {e}")
        raise HTTPException(status_code=502, detail="Weather provider error")

@app.get("/health")
async def health():
    weather_stats = {
        "cache_size": len(_weather_cache),
        "trends_tracked": len(_weather_trends),
        "total_weather_calls": sum(len(trends) for trends in _weather_trends.values())
    }
    
    fashion_stats = {}
    if fashion_engine is not None:
        fashion_stats = {"engine_status": "active", "formality_levels": 4}
    else:
        fashion_stats = {"engine_status": "disabled"}
    
    return {
        "status": "healthy" if engine is not None else "unhealthy",
        "engine_loaded": engine is not None,
        "error": startup_error or None,
        "device": str(engine.device) if engine else None,
        "catalog_size": int(len(engine.items)) if engine and hasattr(engine, 'items') else 0,
        "perf": engine.get_performance_stats() if engine else {},
        "weather_proxy": True,
        "weather_stats": weather_stats,
        "fashion_rules": fashion_stats,
        "enhanced_features": {
            "fashion_rule_enforcement": enhanced_config.ENABLE_FASHION_RULES,
            "formality_classification": fashion_engine is not None
        }
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
    topk: Annotated[int, Form()] = 0,
    weather_hint: Annotated[Optional[str], Form()] = None,
    lat: Annotated[Optional[float], Form()] = None,
    lon: Annotated[Optional[float], Form()] = None,
    weather_json: Annotated[Optional[str], Form()] = None,
    # NEW PARAMETERS for fashion rules
    enable_fashion_rules: Annotated[bool, Form()] = True,
    formality_weight: Annotated[float, Form()] = 0.4,
    strict_rules: Annotated[bool, Form()] = True,
    min_fashion_score: Annotated[float, Form()] = 0.3,
):
    """Enhanced wardrobe recommendation with fashion rule enforcement"""
    if engine is None:
        raise HTTPException(status_code=503, detail=f"Engine not ready: {startup_error}")

    # Normalize & cap per bucket to protect resources
    tops = (tops or [])[:enhanced_config.WARDROBE_CAP]
    bottoms = (bottoms or [])[:enhanced_config.WARDROBE_CAP]
    outerwear = (outerwear or [])[:enhanced_config.WARDROBE_CAP]
    footwear = (footwear or [])[:enhanced_config.WARDROBE_CAP]

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
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats, names

    # Encode per bucket
    feats_map: Dict[str, torch.Tensor] = {}
    names_map: Dict[str, List[str]] = {}

    for bucket, files in cat_lists.items():
        feats, names = await asyncio.get_event_loop().run_in_executor(None, _encode_files, files)
        feats_map[bucket] = feats
        names_map[bucket] = names

    # Weather sources (priority: explicit JSON -> coords -> hint)
    metrics = None
    try:
        if weather_json:
            metrics = json.loads(weather_json)
    except Exception:
        metrics = None

    # If coords provided and no metrics, fetch live weather (non-blocking failure)
    if (lat is not None and lon is not None) and metrics is None:
        try:
            metrics = await _fetch_weather(float(lat), float(lon))
        except Exception as e:
            logger.warning(f"Live weather fetch failed: {e}")

    # Choose final hint
    hint = (weather_hint or "").strip().lower() if weather_hint else None
    if metrics and not hint:
        hint = _hint_from_metrics_enhanced(
            float(metrics.get("temp_c", 0.0)),
            float(metrics.get("precip_mm", 0.0)),
            float(metrics.get("wind_kph", 0.0)),
            int(metrics.get("code", 0)),
            # float(metrics.get("feels_like", metrics.get("temp_c", 0.0))),
            # float(metrics.get("rain_likelihood", 0.0))
        )
    if hint not in {"hot", "cold", "rain", "mild"}:
        hint = "mild"

    weather_context = {
        'hint': hint,
        'temp_c': metrics.get('temp_c', 20) if metrics else 20,
        'metrics': metrics
    }

    # ENHANCED RECOMMENDATION LOGIC
    if enable_fashion_rules and enhanced_recommender is not None:
        logger.info("Using enhanced fashion-rule-aware recommendations")
        
        try:
            # Use the enhanced recommender
            enhanced_combos, rule_analytics = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: enhanced_recommender.enhanced_wardrobe_recommend(
                    cat_lists=cat_lists,
                    feats_map=feats_map,
                    names_map=names_map,
                    weather_context=weather_context,
                    topk=topk,
                    formality_weight=formality_weight,
                    enable_strict_rules=strict_rules
                )
            )
            
            # Convert to response format
            final_items = []
            for combo_data in enhanced_combos:
                score = combo_data['score']
                parts = combo_data['parts']
                explanation = combo_data['explanation']
                
                # Apply minimum fashion score filter
                fashion_score = explanation.get('fashion_rules', {}).get('score', 1.0)
                if fashion_score < min_fashion_score:
                    continue
                
                outfit_parts = [OutfitPart(**part) for part in parts]
                
                final_items.append(OutfitCombo(
                    score=score,
                    parts=outfit_parts,
                    explain=explanation
                ))
            
            logger.info(f"Enhanced recommendation completed: {len(final_items)} combinations")
            return WardrobeResponse(topk=topk, items=final_items)
            
        except Exception as e:
            logger.error(f"Enhanced recommendation failed: {e}")
            # Fall back to original logic below
            enable_fashion_rules = False

    # FALLBACK: Original recommendation logic (if fashion rules disabled or failed)
    logger.info("Using original recommendation logic as fallback")
    
    # Helper functions for original logic
    def _avg_pairwise_cos(indices):
        if len(indices) < 2:
            return 0.8
        
        pairs = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                bi, ii = indices[i]
                bj, ij = indices[j]
                Fi, Fj = feats_map[bi][ii].unsqueeze(0), feats_map[bj][ij].unsqueeze(0)
                cos_sim = float((Fi @ Fj.T).detach().cpu().numpy().item())
                pairs.append(cos_sim)
        
        return min(1.0, (np.mean(pairs) + 1.0) * 0.5)

    def _has(bucket): 
        return feats_map.get(bucket, torch.empty(0)).numel() > 0

    def _basic_weather_component(parts, hint):
        """Basic weather compatibility scoring"""
        if hint == "mild":
            return 1.0
            
        text = " ".join([f"{p.get('slot','')} {p.get('name','')}".lower() for p in parts])
        
        if hint == "hot":
            penalty = 0.3 if any(w in text for w in ["jacket","coat","sweater","hoodie"]) else 0.0
            return max(0.0, 1.0 - penalty)
        elif hint == "cold":
            penalty = 0.4 if any(w in text for w in ["shorts","tank","sleeveless"]) else 0.0
            bonus = 0.2 if any(w in text for w in ["jacket","coat","sweater"]) else 0.0
            return max(0.0, min(1.0, 1.0 - penalty + bonus))
        elif hint == "rain":
            penalty = 0.3 if any(w in text for w in ["suede","canvas"]) else 0.0
            return max(0.0, 1.0 - penalty)
        
        return 1.0

    # Generate base combinations
    combos = []
    bases = []
    
    if _has("tops") and _has("bottoms"):
        tops_feats = feats_map["tops"]
        bottoms_feats = feats_map["bottoms"]
        
        # Create cosine similarity matrix
        sim_matrix = tops_feats @ bottoms_feats.T
        sim_matrix = sim_matrix.detach().cpu().numpy()
        
        for it in range(sim_matrix.shape[0]):
            for ib in range(sim_matrix.shape[1]):
                base_parts = [("tops", it), ("bottoms", ib)]
                compat = _avg_pairwise_cos(base_parts)
                bases.append((compat, base_parts))

    # Create final combinations
    for base_compat, base in bases:
        parts = [
            {"slot": b, "idx": i, "name": names_map[b][i] if i < len(names_map[b]) else f"{b}-{i}"}
            for (b, i) in base
        ]
        
        # Apply weather component
        weather_score = _basic_weather_component(parts, hint)
        final_score = 0.7 * base_compat + 0.3 * weather_score
        
        combos.append((final_score, parts, {
            "compat": round(base_compat, 4),
            "weather": round(weather_score, 4),
            "final": round(final_score, 4),
            "method": "original_fallback",
            "wx_hint": hint
        }))

        # Try adding outerwear if available
        if _has("outerwear"):
            outerwear_feats = feats_map["outerwear"]
            for io in range(outerwear_feats.shape[0]):
                extended_parts = parts + [{"slot": "outerwear", "idx": io, 
                                         "name": names_map["outerwear"][io] if io < len(names_map["outerwear"]) else f"outerwear-{io}"}]
                extended_base = base + [("outerwear", io)]
                extended_compat = _avg_pairwise_cos(extended_base)
                extended_weather = _basic_weather_component(extended_parts, hint)
                extended_score = 0.7 * extended_compat + 0.3 * extended_weather
                
                combos.append((extended_score, extended_parts, {
                    "compat": round(extended_compat, 4),
                    "weather": round(extended_weather, 4),
                    "final": round(extended_score, 4),
                    "method": "original_fallback_with_outerwear",
                    "wx_hint": hint
                }))

        # Try adding footwear if available
        if _has("footwear"):
            footwear_feats = feats_map["footwear"]
            for if_ in range(footwear_feats.shape[0]):
                extended_parts = parts + [{"slot": "footwear", "idx": if_, 
                                         "name": names_map["footwear"][if_] if if_ < len(names_map["footwear"]) else f"footwear-{if_}"}]
                extended_base = base + [("footwear", if_)]
                extended_compat = _avg_pairwise_cos(extended_base)
                extended_weather = _basic_weather_component(extended_parts, hint)
                extended_score = 0.7 * extended_compat + 0.3 * extended_weather
                
                combos.append((extended_score, extended_parts, {
                    "compat": round(extended_compat, 4),
                    "weather": round(extended_weather, 4),
                    "final": round(extended_score, 4),
                    "method": "original_fallback_with_footwear",
                    "wx_hint": hint
                }))

    # Enforce footwear if user added shoes; drop outerwear when it's hot
    has_footwear = feats_map.get("footwear", torch.empty(0)).numel() > 0
    if has_footwear:
        combos = [c for c in combos if any(p.get("slot") == "footwear" for p in c[1])]
    if hint == "hot":
        combos = [c for c in combos if not any(p.get("slot") == "outerwear" for p in c[1])]

    # Sort and limit results
    combos.sort(key=lambda x: x[0], reverse=True)
    final_combos = combos if topk <= 0 else combos[:topk]
    
    # Convert to response format
    final_items = []
    for score, parts, explanation in final_combos:
        outfit_parts = [OutfitPart(**part) for part in parts]
        final_items.append(OutfitCombo(score=score, parts=outfit_parts, explain=explanation))
    
    return WardrobeResponse(topk=topk, items=final_items)

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
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,
    )