from __future__ import annotations
import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset
import colorsys

# Configure logging
logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    logger.warning("ONNXRuntime not available - item classification disabled")

# Enhanced ImageFile settings
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NEUTRALS = {"black", "white", "gray", "grey", "beige", "cream", "denim", "navy", "tan", "brown"}

def _extract_state_dict(checkpoint_obj):
    """Enhanced state dict extraction with better key mapping"""
    if isinstance(checkpoint_obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                sd = checkpoint_obj[key]
                break
        else:
            if any(isinstance(v, torch.Tensor) for v in checkpoint_obj.values()):
                sd = checkpoint_obj
            else:
                raise RuntimeError("Could not locate state_dict in checkpoint")
    else:
        sd = checkpoint_obj

    if sd is None:
        raise RuntimeError("Could not extract state_dict from checkpoint")

    new_sd = {}
    for key, value in sd.items():
        new_key = key
        
        # Remove common prefixes
        if new_key.startswith("module."):
            new_key = new_key[7:]
        if new_key.startswith("model."):
            new_key = new_key[6:]
            
        # Handle backbone mapping
        if new_key.startswith("encoder.backbone."):
            new_key = new_key.replace("encoder.backbone.", "encoder.")
        elif not new_key.startswith("encoder.") and not new_key.startswith("head."):
            if "encoder" not in new_key and "head" not in new_key:
                new_key = f"encoder.{new_key}"
                
        new_sd[new_key] = value

    return new_sd

def bucket_type(category: str) -> str:
    """Optimized bucket type classification"""
    if not category:
        return "other"
    
    cat_lower = category.lower()
    
    footwear_keywords = {"sneaker", "boot", "heel", "flat", "sandal", "loafer", "shoe"}
    outerwear_keywords = {"jacket", "coat", "blazer", "cardigan", "hoodie", "outer"}
    dress_keywords = {"dress", "gown", "jumper dress"}
    bottoms_keywords = {"skirt", "jeans", "trouser", "pant", "short", "legging", "chino"}
    tops_keywords = {"top", "tee", "t-shirt", "shirt", "blouse", "sweater", "pullover", "tank", "camisole"}
    
    if any(kw in cat_lower for kw in footwear_keywords):
        return "footwear"
    elif any(kw in cat_lower for kw in outerwear_keywords):
        return "outerwear"
    elif any(kw in cat_lower for kw in dress_keywords):
        return "dress"
    elif any(kw in cat_lower for kw in bottoms_keywords):
        return "bottoms"
    elif any(kw in cat_lower for kw in tops_keywords):
        return "tops"
    
    return "other"

def load_image_preproc(img: Image.Image, img_size: int) -> torch.Tensor:
    """Image preprocessing - exact match with script"""
    # Use BICUBIC resampling like script
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.Resampling.BICUBIC)
    
    arr = np.asarray(img, dtype=np.float32)
    arr = arr / 255.0
    
    # Use exact same normalization constants
    arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    
    return torch.from_numpy(arr)

class ImageOnlyCompatibility(torch.nn.Module):
    """Optimized compatibility model with better memory usage"""
    
    def __init__(self, backbone="efficientnet_b0", embed_dim=512):
        super().__init__()
        import timm
        
        self.embed_dim = embed_dim
        self.encoder = timm.create_model(
            backbone, 
            pretrained=False, 
            num_classes=embed_dim,
            drop_rate=0.0
        )
        
        self.head = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 1),
        )
    
    def forward(self, xa, xb):
        fa = self.encoder(xa)
        fb = self.encoder(xb)
        x = torch.cat([fa, fb], dim=1)
        return self.head(x).squeeze(1)

class OptimizedCatalogDS(Dataset):
    """Optimized catalog dataset with caching"""
    
    def __init__(self, items_df: pd.DataFrame, images_root: Path, img_size: int):
        self.df = items_df.reset_index(drop=True)
        self.root = Path(images_root)
        self.img_size = img_size
        self._cache = {}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        item_id = str(row["item_id"])
        image_path = str(row["image_path"])
        
        cache_key = f"{item_id}_{self.img_size}"
        if cache_key in self._cache:
            tensor = self._cache[cache_key]
        else:
            try:
                img = Image.open(self.root / image_path).convert("RGB")
                tensor = load_image_preproc(img, self.img_size)
                
                if len(self.df) < 1000:
                    self._cache[cache_key] = tensor
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                tensor = torch.zeros(3, self.img_size, self.img_size)
        
        return (
            item_id,
            image_path,
            str(row["category"]),
            str(row["title"]),
            tensor
        )

class RecommendEngine:
    """Fixed recommendation engine with proper feature alignment and color detection"""
    
    def __init__(
        self,
        ckpt_path: Path,
        catalog_csv: Path,
        images_root: Path,
        items_aug_csv: Optional[Path] = None,
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 2,
        device: Optional[torch.device] = None,
        item_cls_onnx: Optional[Path] = None,
        item_cls_meta: Optional[Path] = None,
        topn_rerank: int = None,
        feats_cache: Optional[Path] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = max(0, num_workers)
        self.topn_rerank = int(os.getenv("OUTFIT_TOPN_RERANK", str(topn_rerank or 4000)))
        cache_env = os.getenv("OUTFIT_FEATS_CACHE", "")
        self.feats_cache_path = Path(feats_cache or cache_env) if (feats_cache or cache_env) else None
        
        logger.info(f"Initializing RecommendEngine on {self.device}")
        
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Initialize components
        self._load_model(ckpt_path)
        self._load_catalog_data(catalog_csv, images_root, items_aug_csv)
        self._load_item_classifier(item_cls_onnx, item_cls_meta)
        self._optimize_model()
        
        logger.info("RecommendEngine initialization complete")
    
    def _load_model(self, ckpt_path: Path):
        """Load and initialize the compatibility model"""
        logger.info(f"Loading checkpoint from {ckpt_path}")
        
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
            
            backbone = meta.get("arch", "efficientnet_b0")
            embed_dim = meta.get("embed_dim", 512)
            
            logger.info(f"Model config: backbone={backbone}, embed_dim={embed_dim}")
            
            self.model = ImageOnlyCompatibility(backbone=backbone, embed_dim=embed_dim)
            
            state_dict = _extract_state_dict(checkpoint)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            # Decide if the learned head is actually usable
            self._head_ok = True
            try:
                miss = [k for k in missing_keys if k.startswith("head.")]
                self._head_ok = (len(miss) == 0)
            except NameError:
                # if you didn't keep 'missing_keys', be conservative
                self._head_ok = False

            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)} (showing first 5: {missing_keys[:5]})")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)} (showing first 5: {unexpected_keys[:5]})")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.catalog_feats = None
            self.catalog_feats_norm = None
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_catalog_data(self, catalog_csv: Path, images_root: Path, items_aug_csv: Optional[Path]):
        """Load and process catalog data with proper verification"""
        logger.info(f"Loading catalog from {catalog_csv}")
        
        try:
            # Load main catalog
            self.items = pd.read_csv(
                catalog_csv,
                dtype={"item_id": str, "image_path": str, "title": str, "category": str},
                keep_default_na=False,
                na_values=[]
            ).fillna("")
            
            # Ensure required columns
            for col in ["item_id", "image_path", "title", "category"]:
                if col not in self.items.columns:
                    self.items[col] = ""
            
            # Add bucket column with debug info
            self.items["bucket"] = self.items["category"].apply(bucket_type)
            
            # Debug: Print bucket distribution
            bucket_counts = self.items["bucket"].value_counts()
            logger.info(f"Bucket distribution: {bucket_counts.to_dict()}")
            
            self.images_root = Path(images_root)
            
            logger.info(f"Loaded {len(self.items)} catalog items")
            
            # Load color data
            self.color_lookup = {}
            if items_aug_csv and Path(items_aug_csv).exists():
                self._load_color_data(items_aug_csv)
            
            # Load feature cache if available
            self.catalog_feats = None
            self.catalog_feats_norm = None
            if self.feats_cache_path and self.feats_cache_path.exists():
                self._try_load_feats_cache()
            else:
                logger.warning("No feature cache available - will compute features on demand")

        except Exception as e:
            logger.error(f"Failed to load catalog data: {e}")
            raise
    
    def _try_load_feats_cache(self):
        """Load and align precomputed features with proper validation"""
        try:
            logger.info(f"Loading feature cache from {self.feats_cache_path}")
            obj = torch.load(self.feats_cache_path, map_location="cpu")
            
            cached_feats = obj["feats"].float()
            cached_paths = [str(p) for p in obj["paths"]]
            
            # Validate cache compatibility
            cache_embed_dim = int(obj.get("embed_dim", cached_feats.shape[1]))
            cache_img_size = int(obj.get("img_size", self.img_size))
            
            model_embed_dim = self.model.embed_dim
            
            if cache_embed_dim != model_embed_dim:
                logger.warning(f"Cache embed_dim {cache_embed_dim} != model {model_embed_dim}, ignoring cache")
                return
            
            if cache_img_size != self.img_size:
                logger.warning(f"Cache img_size {cache_img_size} != current {self.img_size}, ignoring cache")
                return
            
            # Build alignment mapping
            current_paths = [str(p) for p in self.items["image_path"]]
            path_to_cache_idx = {p: i for i, p in enumerate(cached_paths)}
            
            aligned_indices = []
            valid_item_indices = []
            
            for item_idx, path in enumerate(current_paths):
                cache_idx = path_to_cache_idx.get(path)
                if cache_idx is not None:
                    aligned_indices.append(cache_idx)
                    valid_item_indices.append(item_idx)
            
            if len(aligned_indices) == 0:
                logger.warning("No matching paths between cache and current catalog")
                return
            
            # Extract aligned features
            aligned_feats = cached_feats[aligned_indices]
            
            # Update items to only include cached ones
            self.items = self.items.iloc[valid_item_indices].reset_index(drop=True)
            
            # Store features
            self.catalog_feats = aligned_feats.contiguous()
            self.catalog_feats_norm = F.normalize(aligned_feats, dim=1)
            
            logger.info(f"Loaded aligned feature cache: {self.catalog_feats.shape} features for {len(self.items)} items")
            
        except Exception as e:
            logger.error(f"Failed to load feature cache: {e}")
            self.catalog_feats = None
            self.catalog_feats_norm = None
    
    def _load_color_data(self, items_aug_csv: Path):
        """Load color augmentation data with improved color detection"""
        try:
            aug = pd.read_csv(items_aug_csv, dtype=str, keep_default_na=False)
            
            required_cols = ["primary_name", "primary_hex", "primary_frac",
                           "secondary_name", "secondary_hex", "secondary_frac"]
            for col in required_cols:
                if col not in aug.columns:
                    aug[col] = ""
            
            color_data = {}
            for _, row in aug.iterrows():
                image_path = str(row.get("image_path", ""))
                if image_path:
                    color_name, frac = self._extract_color_info_improved(row)
                    color_data[image_path] = (color_name, frac)
            
            self.color_lookup = color_data
            logger.info(f"Loaded color data for {len(self.color_lookup)} items")
            
        except Exception as e:
            logger.warning(f"Failed to load color data: {e}")
            self.color_lookup = {}
    
    def _extract_color_info_improved(self, row) -> Tuple[str, float]:
        """Improved color extraction with better fallback logic"""
        # Try primary color first
        primary_name = self._normalize_color_name(row.get("primary_name", ""))
        primary_hex = row.get("primary_hex", "")
        
        try:
            primary_frac = float(row.get("primary_frac", "0.0"))
        except (ValueError, TypeError):
            primary_frac = 0.0
        
        # Try secondary color
        secondary_name = self._normalize_color_name(row.get("secondary_name", ""))
        secondary_hex = row.get("secondary_hex", "")
        
        try:
            secondary_frac = float(row.get("secondary_frac", "0.0"))
        except (ValueError, TypeError):
            secondary_frac = 0.0
        
        # Choose best color
        color_name = ""
        frac = 0.0
        
        # Prefer primary if it has reasonable fraction and is not just "white"
        if primary_name and primary_frac > 0.1 and primary_name != "white":
            color_name = primary_name
            frac = primary_frac
        elif primary_hex and primary_frac > 0.1:
            hex_color = self._hex_to_family(primary_hex)
            if hex_color and hex_color != "white":
                color_name = hex_color
                frac = primary_frac
        
        # Fallback to secondary
        if not color_name:
            if secondary_name and secondary_frac > 0.1:
                color_name = secondary_name
                frac = secondary_frac
            elif secondary_hex and secondary_frac > 0.1:
                color_name = self._hex_to_family(secondary_hex)
                frac = secondary_frac
        
        # Final fallback
        if not color_name:
            color_name = "neutral"
            frac = 1.0
        
        return color_name, max(0.0, min(1.0, frac))
    
    def _load_item_classifier(self, item_cls_onnx: Optional[Path], item_cls_meta: Optional[Path]):
        """Load ONNX item classifier"""
        self.cls = None
        self.cls_labels = []
        self.cls_img_size = 224
        
        if not item_cls_onnx or not Path(item_cls_onnx).exists() or ort is None:
            logger.info("ONNX classifier not available")
            return
        
        try:
            providers = ["CPUExecutionProvider"]
            if torch.cuda.is_available():
                providers.insert(0, "CUDAExecutionProvider")
            
            self.cls = ort.InferenceSession(str(item_cls_onnx), providers=providers)
            
            if item_cls_meta and Path(item_cls_meta).exists():
                with open(item_cls_meta, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    self.cls_labels = meta.get("class_names", [])
                    self.cls_img_size = int(meta.get("img_size", 224))
            
            self._label_to_bucket = {
                "jeans": "bottoms",
                "trousers": "bottoms", 
                "shorts": "bottoms",
                "leggings": "bottoms",
                "skirt": "bottoms",
                "shirt": "tops",
                "blouse": "tops",
                "t-shirt": "tops",
                "sweater": "tops",
                "jacket": "outerwear",
                "coat": "outerwear",
                "blazer": "outerwear",
                "sneakers": "footwear",
                "boots": "footwear",
                "heels": "footwear",
            }
            
            logger.info(f"ONNX classifier loaded: {len(self.cls_labels)} classes, img_size={self.cls_img_size}")
            
        except Exception as e:
            logger.warning(f"Failed to load ONNX classifier: {e}")
            self.cls = None
    
    def _optimize_model(self):
        """Optimize model for inference"""
        try:
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
            
            self._warmup_model()
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
    
    def _warmup_model(self):
        """Warmup model with dummy inference"""
        try:
            dummy_input = torch.randn(1, 3, self.img_size, self.img_size, device=self.device)
            with torch.no_grad():
                _ = self.model.encoder(dummy_input)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    # Color utility methods
    def _normalize_color_name(self, name: str) -> str:
        if not isinstance(name, str):
            return ""
        normalized = name.strip().lower()
        return {"grey": "gray", "navy blue": "navy"}.get(normalized, normalized)
    
    def _hex_to_rgb(self, hex_str: str) -> Optional[Tuple[int, int, int]]:
        if not isinstance(hex_str, str) or not hex_str:
            return None
        s = hex_str.strip().lstrip("#")
        if len(s) == 3:
            s = "".join([ch*2 for ch in s])
        if len(s) != 6:
            return None
        try:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        except ValueError:
            return None
    
    def _rgb_to_hsv(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        r_, g_, b_ = r/255.0, g/255.0, b/255.0
        mx, mn = max(r_, g_, b_), min(r_, g_, b_)
        diff = mx - mn
        
        if diff == 0:
            h = 0
        elif mx == r_:
            h = (60 * ((g_-b_) / diff) + 360) % 360
        elif mx == g_:
            h = (60 * ((b_-r_) / diff) + 120) % 360
        else:
            h = (60 * ((r_-g_) / diff) + 240) % 360
            
        s = 0 if mx == 0 else diff / mx
        v = mx
        return h, s, v
    
    def _hsv_to_family(self, h: float, s: float, v: float) -> str:
        """HSV to color family mapping - exact match with script"""
        if v < 0.18:
            return "black"
        if s < 0.10:
            if v > 0.85:
                return "white"
            return "gray"
        
        # Exact hue ranges from script
        if h < 15 or h >= 345:
            return "red"
        if h < 45:
            return "orange"  
        if h < 70:
            return "yellow"
        if h < 170:
            return "green"
        if h < 200:
            return "teal"
        if h < 225:
            return "cyan"
        if h < 255:
            return "blue"
        if h < 275:
            return "navy"
        if h < 300:
            return "purple"
        if h < 345:
            return "pink"
        return "other"
    
    def _hex_to_family(self, hex_str: str) -> str:
        rgb = self._hex_to_rgb(hex_str)
        if rgb is None:
            return ""
        h, s, v = self._rgb_to_hsv(*rgb)
        return self._hsv_to_family(h, s, v)
    
    def _dominant_color_kmeans(self, image: Image.Image) -> str:
        """K-means color extraction matching script behavior"""
        try:
            # Use same parameters as script
            im = image.convert("RGB").resize((64, 64), Image.Resampling.BICUBIC)
            X = np.asarray(im, dtype=np.float32).reshape(-1, 3)
            
            # Simple k-means implementation like in script
            k = 3
            iters = 8
            
            # Initialize with random samples
            np.random.seed(42)  # For consistency
            idx = np.random.choice(X.shape[0], size=k, replace=False)
            C = X[idx].copy()
            
            for _ in range(iters):
                # Assign points to clusters
                d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
                lab = d2.argmin(axis=1)
                
                # Update cluster centers
                for j in range(k):
                    pts = X[lab == j]
                    if len(pts):
                        C[j] = pts.mean(axis=0)
            
            # Pick largest cluster
            counts = np.bincount(lab, minlength=k)
            j = int(counts.argmax())
            r, g, b = C[j]
            h, s, v = self._rgb_to_hsv(int(r), int(g), int(b))
            return self._hsv_to_family(h, s, v)
            
        except Exception as e:
            logger.debug(f"K-means color extraction failed: {e}")
            return self._dominant_color_fast(image)
        
    def _dominant_color_fast(self, image: Image.Image) -> str:
        """Improved fast dominant color extraction matching the script logic"""
        try:
            # Use same resize as script for consistency
            img = image.convert("RGB").resize((48, 48), Image.Resampling.BICUBIC)
            arr = np.asarray(img, dtype=np.uint8)
            
            # Convert all pixels to HSV like the script does
            hsv_values = []
            for y in range(arr.shape[0]):
                for x in range(arr.shape[1]):
                    r, g, b = arr[y, x]
                    h, s, v = self._rgb_to_hsv(int(r), int(g), int(b))
                    hsv_values.append((h, s, v))
            
            hsv_array = np.array(hsv_values)
            
            # Filter by saturation like the script
            mask = hsv_array[:, 1] >= 0.1  # s >= 0.1
            if mask.sum() == 0:
                # Robust fallback: vote by extremes instead of using the mean
                v = hsv_array[:, 2]
                black_ratio = (v < 0.35).mean()
                white_ratio = (v > 0.85).mean()
                if black_ratio > white_ratio and black_ratio > 0.15:
                    return "black"
                if white_ratio >= black_ratio and white_ratio > 0.15:
                    return "white"
                return "gray"

            else:
                # Use median hue from saturated pixels
                h_median = float(np.median(hsv_array[mask, 0]))
                s_median = float(np.median(hsv_array[mask, 1]))
                v_median = float(np.median(hsv_array[mask, 2]))
                return self._hsv_to_family(h_median, s_median, v_median)
                
        except Exception as e:
            logger.debug(f"Fast color extraction failed: {e}")
            return "unknown"
        
    def _color_compatibility_score(self, anchor_color: str, candidate_color: str) -> float:
        """Color compatibility score - exact match with script"""
        anchor = self._normalize_color_name(anchor_color)
        candidate = self._normalize_color_name(candidate_color)
        
        if not anchor or not candidate:
            return 0.0
        if anchor == candidate:
            return 0.5
        if anchor in NEUTRALS or candidate in NEUTRALS:
            return 0.3
            
        # Use exact complement mapping from script
        COMPLEMENT = {
            "red": {"green", "olive", "teal"},
            "green": {"magenta", "purple", "red"},
            "blue": {"orange", "tan", "brown"},
            "orange": {"blue", "navy"},
            "yellow": {"purple", "violet"},
            "purple": {"yellow", "beige"},
            "pink": {"olive", "army", "green"},
            "teal": {"maroon", "red"},
        }
        
        if anchor in COMPLEMENT and candidate in COMPLEMENT[anchor]:
            return 0.4
        return 0.0
    
    @torch.no_grad()
    def predict_anchor_bucket(self, image: Image.Image) -> str:
        """Predict clothing bucket using improved heuristics"""
        try:
            # Use classifier if available
            bucket = self.predict_bucket_from_classifier(image, conf_thresh=0.3)
            if bucket:
                return bucket
            
            # Fallback to basic heuristics
            width, height = image.size
            aspect_ratio = width / height
            
            # Improved heuristics
            if aspect_ratio > 1.8:  # Very wide items
                return "tops"
            elif aspect_ratio < 0.6:  # Very tall items  
                return "outerwear"
            elif 0.8 <= aspect_ratio <= 1.2:  # Square-ish items
                return "tops"
            else:
                return "bottoms"
                
        except Exception as e:
            logger.debug(f"Anchor bucket prediction failed: {e}")
            return "unknown"
    
    @torch.no_grad() 
    def predict_bucket_from_classifier(self, image: Image.Image, conf_thresh: float = 0.5) -> Optional[str]:
        """Predict clothing bucket using ONNX classifier"""
        if self.cls is None:
            return None
            
        try:
            img_resized = image.convert("RGB").resize((self.cls_img_size, self.cls_img_size), Image.Resampling.BICUBIC)
            arr = np.asarray(img_resized, dtype=np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            arr = np.transpose(arr, (2, 0, 1))[None, ...]
            
            input_name = self.cls.get_inputs()[0].name
            output_name = self.cls.get_outputs()[0].name
            logits = self.cls.run([output_name], {input_name: arr})[0]
            
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            
            if confidence >= conf_thresh and pred_idx < len(self.cls_labels):
                label = self.cls_labels[pred_idx]
                bucket = self._label_to_bucket.get(label.lower())
                if bucket:
                    logger.debug(f"Classified as {label} -> {bucket} (conf: {confidence:.3f})")
                    return bucket
                    
        except Exception as e:
            logger.debug(f"Classifier prediction failed: {e}")
            
        return None
    
    @torch.no_grad()
    def recommend(
        self,
        user_image: Image.Image,
        *,
        allow_types: str = "tops,bottoms,outerwear,footwear",
        per_bucket: int = 5,
        topk: int = 20,
        color_weight: float = 0.1,
        anchor_color_mode: str = "auto",
        filter_same_bucket: bool = True,
    ) -> Tuple[str, pd.DataFrame]:
        """Generate catalog recommendations with exact script matching"""
        
        # Set seed for consistency
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Debug: Check color lookup availability
        logger.info(f"Color lookup available: {len(self.color_lookup) > 0}, entries: {len(self.color_lookup)}")
        
        # Get anchor color using the same logic as script
        anchor_color = ""
        
        # Try to get from color lookup first (auto mode)
        if anchor_color_mode == "auto" and self.color_lookup:
            # This would need the image path, which we don't have for user uploads
            # So auto mode falls back to HSV
            pass
        
        # Compute anchor color based on mode
        if not anchor_color or anchor_color_mode == "km":
            anchor_color = self._dominant_color_kmeans(user_image)
        elif anchor_color_mode == "hsv" or anchor_color_mode == "auto":
            anchor_color = self._dominant_color_fast(user_image)
        
        logger.info(f"Detected anchor color: {anchor_color} (mode: {anchor_color_mode})")
        
        # Detect anchor bucket
        anchor_bucket = self.predict_anchor_bucket(user_image)
        logger.info(f"Detected anchor bucket: {anchor_bucket}")
        
        # Filter catalog with improved logic
        items = self._filter_catalog_improved(allow_types, anchor_bucket, filter_same_bucket)
        
        if len(items) == 0:
            logger.warning("No items found after filtering")
            return anchor_color, pd.DataFrame()
        
        logger.info(f"Found {len(items)} items after filtering")
        
        # Encode user image once
        user_tensor = load_image_preproc(user_image, self.img_size).unsqueeze(0).to(self.device)
        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            user_features = self.model.encoder(user_tensor)
        
        # Score items
        if self.catalog_feats is not None:
            scores = self._score_with_cache(items, user_features, anchor_color, color_weight)
        else:
            scores = self._score_catalog_items(items, user_features, anchor_color, color_weight)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'item_id': items['item_id'].values,
            'image_path': items['image_path'].values, 
            'category': items['category'].values,
            'title': items['title'].values,
            'bucket': items['bucket'].values,
            'score': scores
        })
        
        # Diversify and rank
        final_results = self._diversify_results(results_df, per_bucket, topk)
        
        return anchor_color, final_results
    
    def _filter_catalog_improved(self, allow_types: str, anchor_bucket: str, filter_same_bucket: bool) -> pd.DataFrame:
        """Improved catalog filtering with better bucket handling"""
        filtered = self.items.copy()
        
        # Debug: Check original distribution
        logger.debug(f"Original catalog size: {len(filtered)}")
        logger.debug(f"Bucket distribution: {filtered['bucket'].value_counts().to_dict()}")
        
        # Filter by allowed types
        if allow_types:
            allowed = set(t.strip().lower() for t in allow_types.split(",") if t.strip())
            if allowed:
                filtered = filtered[filtered["bucket"].isin(allowed)]
                logger.debug(f"After allow_types filter: {len(filtered)} items")
        
        # Filter same bucket if requested and anchor bucket is valid
        if filter_same_bucket and anchor_bucket in {"tops", "bottoms", "outerwear", "footwear"}:
            original_len = len(filtered)
            filtered = filtered[filtered["bucket"] != anchor_bucket]
            logger.debug(f"Filtered out {anchor_bucket}: {original_len} -> {len(filtered)} items")
        
        return filtered.reset_index(drop=True)
    
    def _score_with_cache(self, items: pd.DataFrame, user_features: torch.Tensor, anchor_color: str, color_weight: float) -> np.ndarray:
        """Score items using precomputed features with proper color bonus"""
        try:
            # Get original indices from items
            original_indices = items.index.to_numpy()
            
            if len(original_indices) == 0:
                return np.array([])
            
            fb = self.catalog_feats[original_indices].to(self.device, non_blocking=True)
            
            with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                if getattr(self, "_head_ok", False):
                    ua = user_features.repeat(fb.shape[0], 1)
                    logits = self.model.head(torch.cat([ua, fb], dim=1)).squeeze(1)
                    scores = torch.sigmoid(logits)
                else:
                    # cosine fallback using cached features
                    ua = F.normalize(user_features.repeat(fb.shape[0], 1), dim=1)
                    fb_n = F.normalize(fb, dim=1)
                    scores = (F.cosine_similarity(ua, fb_n, dim=1) + 1.0) * 0.5
                scores = scores.detach().cpu().numpy()
            
            # Apply color bonus - THIS IS THE KEY FIX
            if anchor_color and color_weight > 0 and self.color_lookup:
                for i, (_, row) in enumerate(items.iterrows()):
                    img_path = str(row["image_path"])
                    cname, frac = self.color_lookup.get(img_path, ("", 1.0))
                    
                    # Use exact same logic as script
                    bonus = color_weight * self._color_compatibility_score(anchor_color, cname) * max(0.5, float(frac or 1.0))
                    scores[i] = float(scores[i]) + bonus
                    
                    # Debug logging for first few items
                    if i < 3:
                        logger.info(f"Item {i}: path={img_path}, color={cname}, frac={frac}, bonus={bonus}, final_score={scores[i]}")
            
            return scores
            
        except Exception as e:
            logger.error(f"Cache scoring failed: {e}, falling back to slow path")
            return self._score_catalog_items(items, user_features, anchor_color, color_weight)
        
    def _score_catalog_items(self, items: pd.DataFrame, user_features: torch.Tensor, anchor_color: str, color_weight: float) -> np.ndarray:
        """Score catalog items efficiently without cache"""
        if len(items) == 0:
            return np.array([])
        
        dataset = OptimizedCatalogDS(items, self.images_root, self.img_size)
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(items)),
            shuffle=False,
            num_workers=min(self.num_workers, len(items) // 10 + 1),
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(self.num_workers > 0 and len(items) > 100),
        )
        
        all_scores = []
        
        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            for batch_data in dataloader:
                item_ids, image_paths, categories, titles, item_tensors = batch_data
                item_tensors = item_tensors.to(self.device, non_blocking=True)
                
                # Get item features
                item_features = self.model.encoder(item_tensors)
                batch_size = item_features.shape[0]
                
                # Compute compatibility scores (fallback to cosine if head not loaded)
                if getattr(self, "_head_ok", False):
                    user_repeated = user_features.repeat(batch_size, 1)
                    combined = torch.cat([user_repeated, item_features], dim=1)
                    logits = self.model.head(combined).squeeze(1)
                    scores = torch.sigmoid(logits)
                else:
                    # cosine similarity in [-1,1] → [0,1]
                    scores = F.cosine_similarity(
                        user_features.repeat(batch_size, 1), item_features, dim=1
                    )
                    scores = (scores + 1.0) * 0.5


                # Apply color bonus - FIXED VERSION
                if color_weight > 0 and anchor_color and self.color_lookup:
                    for i, img_path in enumerate(image_paths):
                        cname, frac = self.color_lookup.get(str(img_path), ("", 1.0))
                        bonus = color_weight * self._color_compatibility_score(anchor_color, cname) * max(0.5, float(frac or 1.0))
                        scores[i] = scores[i] + bonus
                
                all_scores.extend(scores)
        
        return np.array(all_scores)

    def _diversify_results(self, results_df: pd.DataFrame, per_bucket: int, topk: int) -> pd.DataFrame:
        """Diversify results by bucket and return top-k with better distribution"""
        if len(results_df) == 0:
            return pd.DataFrame()
        
        diversified_pieces = []
        
        # Group by bucket and get top items from each
        for bucket, group in results_df.groupby("bucket"):
            bucket_results = group.sort_values("score", ascending=False).head(per_bucket)
            diversified_pieces.append(bucket_results)
            logger.debug(f"Selected {len(bucket_results)} items from {bucket} bucket")
        
        if diversified_pieces:
            final_df = pd.concat(diversified_pieces, ignore_index=True)
            final_df = final_df.sort_values("score", ascending=False).head(topk)
            logger.info(f"Final results: {len(final_df)} items across {final_df['bucket'].nunique()} buckets")
            return final_df.reset_index(drop=True)
        
        return pd.DataFrame()
    
    def wardrobe_recommend(
        self,
        tops: List = None,
        bottoms: List = None, 
        outerwear: List = None,
        footwear: List = None,
        topk: int = 10
    ) -> Dict:
        """Generate wardrobe outfit combinations with improved logic"""
        
        # Validate inputs
        tops = tops or []
        bottoms = bottoms or []
        outerwear = outerwear or []
        footwear = footwear or []
        
        # Check minimum requirements
        categories_with_items = sum([len(tops) > 0, len(bottoms) > 0, len(outerwear) > 0, len(footwear) > 0])
        if categories_with_items < 2:
            raise ValueError("Need at least 2 categories with items")
        
        # Encode all wardrobe items
        encoded_items = {}
        item_names = {}
        
        orig_index_map = {}
        for category, files in [("tops", tops), ("bottoms", bottoms), ("outerwear", outerwear), ("footwear", footwear)]:
            if files:
                features, names, orig_idx = self._encode_wardrobe_files(files)
                if len(features) > 0:
                    encoded_items[category] = features
                    item_names[category] = names
                    orig_index_map[category] = orig_idx
        
        logger.info(f"Encoded wardrobe: {[(k, v.shape[0]) for k, v in encoded_items.items()]}")
        
        # Generate combinations with improved algorithm
        outfits = self._generate_outfit_combinations_improved(encoded_items, item_names, orig_index_map, topk)
        
        return {"topk": topk, "items": outfits}
    
    @torch.no_grad()
    def _encode_wardrobe_files(self, files: List) -> Tuple[torch.Tensor, List[str], List[int]]:
        """Encode wardrobe files efficiently and keep original indices"""
        features_list: List[torch.Tensor] = []
        names_list: List[str] = []
        orig_indices: List[int] = []
        
        for i, file in enumerate(files):
            try:
                if hasattr(file, 'file'):
                    # UploadFile object
                    image = Image.open(file.file).convert("RGB")
                    name = getattr(file, 'filename', f'item_{i}')
                else:
                    # Assume PIL Image
                    image = file
                    name = f'image_{i}'
                
                # Preprocess and encode
                tensor = load_image_preproc(image, self.img_size).unsqueeze(0).to(self.device)
                
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    features = self.model.encoder(tensor)
                
                features_list.append(features.squeeze(0))
                names_list.append(name)
                orig_indices.append(i)
                
            except Exception as e:
                logger.warning(f"Failed to encode wardrobe item {i}: {e}")
                continue
        
        if features_list:
            return torch.stack(features_list), names_list, orig_indices
        else:
            return torch.empty((0, self.model.embed_dim), device=self.device), [], []
    
    def _generate_outfit_combinations_improved(self, encoded_items: Dict, item_names: Dict, orig_index_map: Dict, topk: int) -> List[Dict]:
        """Generate outfit combinations with improved algorithm"""
        outfits = []
        categories = list(encoded_items.keys())
        
        if len(categories) < 2:
            return []
        
        logger.info(f"Generating combinations from categories: {categories}")
        
        # Generate all possible combinations between categories
        all_combinations = []
        
        # Generate pairs first (most important)
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat1, cat2 = categories[i], categories[j]
                scores = self._compute_pairwise_scores(encoded_items[cat1], encoded_items[cat2])
                
                for idx1 in range(len(encoded_items[cat1])):
                    for idx2 in range(len(encoded_items[cat2])):
                        score = float(scores[idx1, idx2])
                        combination = {
                            "score": score,
                            "parts": [
                                {"slot": cat1, "idx": int(orig_index_map[cat1][idx1]), "name": item_names[cat1][idx1]},
                                {"slot": cat2, "idx": int(orig_index_map[cat2][idx2]), "name": item_names[cat2][idx2]},
                            ]
                        }
                        all_combinations.append(combination)
        
        # Generate three-way combinations if we have 3+ categories
        if len(categories) >= 3:
            for i in range(len(categories)):
                for j in range(i + 1, len(categories)):
                    for k in range(j + 1, len(categories)):
                        cat1, cat2, cat3 = categories[i], categories[j], categories[k]
                        
                        # Limit combinations for performance
                        max_items_per_cat = 3
                        
                        for idx1 in range(min(len(encoded_items[cat1]), max_items_per_cat)):
                            for idx2 in range(min(len(encoded_items[cat2]), max_items_per_cat)):
                                for idx3 in range(min(len(encoded_items[cat3]), max_items_per_cat)):
                                    # Compute three-way compatibility
                                    score1 = self._compute_pairwise_scores(
                                        encoded_items[cat1][idx1:idx1+1], 
                                        encoded_items[cat2][idx2:idx2+1]
                                    )[0, 0]
                                    score2 = self._compute_pairwise_scores(
                                        encoded_items[cat1][idx1:idx1+1], 
                                        encoded_items[cat3][idx3:idx3+1]
                                    )[0, 0]
                                    score3 = self._compute_pairwise_scores(
                                        encoded_items[cat2][idx2:idx2+1], 
                                        encoded_items[cat3][idx3:idx3+1]
                                    )[0, 0]
                                    
                                    # Average the pairwise scores
                                    avg_score = float((score1 + score2 + score3) / 3)
                                    
                                    combination = {
                                        "score": avg_score,
                                        "parts": [
                                            {"slot": cat1, "idx": int(orig_index_map[cat1][idx1]), "name": item_names[cat1][idx1]},
                                            {"slot": cat2, "idx": int(orig_index_map[cat2][idx2]), "name": item_names[cat2][idx2]},
                                            {"slot": cat3, "idx": int(orig_index_map[cat3][idx3]), "name": item_names[cat3][idx3]},
                                        ]
                                    }
                                    all_combinations.append(combination)
        
        # Sort by score and return top combinations
        all_combinations.sort(key=lambda x: x["score"], reverse=True)
        outfits = all_combinations[:topk]
        
        logger.info(f"Generated {len(outfits)} outfit combinations")
        
        return outfits
    
    def _compute_pairwise_scores(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise compatibility scores between two feature sets"""
        n1, n2 = len(features1), len(features2)
        if n1 == 0 or n2 == 0:
            return torch.empty((n1, n2), device=self.device)

        if getattr(self, "_head_ok", False):
            scores = torch.zeros(n1, n2, device=self.device)
            chunk_size = min(32, n1)
            for i in range(0, n1, chunk_size):
                end_i = min(i + chunk_size, n1)
                f1 = features1[i:end_i]                  # [c,D]
                expanded_f1 = f1.unsqueeze(1).repeat(1, n2, 1)
                expanded_f2 = features2.unsqueeze(0).repeat(end_i - i, 1, 1)
                combined = torch.cat([expanded_f1, expanded_f2], dim=-1).view(-1, features1.shape[1]*2)
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    logits = self.model.head(combined)
                    chunk_scores = torch.sigmoid(logits).view(end_i - i, n2)
                scores[i:end_i] = chunk_scores
            return scores
        else:
            # cosine fallback
            f1 = F.normalize(features1, dim=1)
            f2 = F.normalize(features2, dim=1)
            return torch.clamp(f1 @ f2.T, -1.0, 1.0).mul_(0.5).add_(0.5)

    def debug_color_data(self, sample_paths: List[str] = None):
        """Debug method to check color data"""
        if not self.color_lookup:
            logger.warning("No color lookup data available")
            return
        
        logger.info(f"Color lookup contains {len(self.color_lookup)} entries")
        
        # Sample some entries
        sample_items = list(self.color_lookup.items())[:5]
        for path, (color, frac) in sample_items:
            logger.info(f"Color data sample: {path} -> color={color}, frac={frac}")
        
        # Check specific paths if provided
        if sample_paths:
            for path in sample_paths:
                color_data = self.color_lookup.get(path, ("NOT_FOUND", 0.0))
                logger.info(f"Path {path}: {color_data}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)