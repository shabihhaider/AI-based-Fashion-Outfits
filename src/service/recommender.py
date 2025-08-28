from __future__ import annotations
import os
import json
import logging
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
import torch.nn.functional as F
from contextlib import contextmanager

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import colorsys
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import cv2

# Enhanced imports with error handling
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced ImageFile settings
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Constants with scientific backing
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Enhanced color analysis constants
NEUTRALS = {
    "black", "white", "gray", "grey", "beige", "cream", "navy", 
    "tan", "brown", "charcoal", "ivory", "khaki", "taupe"
}

# Color harmony rules based on color theory
COMPLEMENTARY_PAIRS = {
    "red": ["green", "teal", "mint"],
    "blue": ["orange", "coral", "peach"],
    "yellow": ["purple", "violet", "indigo"],
    "green": ["red", "magenta", "pink"],
    "orange": ["blue", "navy", "cyan"],
    "purple": ["yellow", "gold", "lime"]
}

# Performance configuration
@dataclass
class PerformanceConfig:
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    compile_model: bool = True
    cache_size: int = 1000
    faiss_nprobe: int = 32
    max_memory_gb: float = 8.0

def get_optimal_config() -> PerformanceConfig:
    """Automatically configure performance based on hardware"""
    config = PerformanceConfig()
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        config.max_memory_gb = gpu_memory * 0.8  # Use 80% of GPU memory
        config.batch_size = min(64, int(gpu_memory // 2))
        config.num_workers = min(8, os.cpu_count())
        config.mixed_precision = True
        config.compile_model = torch.cuda.get_device_capability()[0] >= 7  # V100+
    else:
        config.batch_size = 16
        config.num_workers = min(4, os.cpu_count())
        config.mixed_precision = False
        config.compile_model = False
    
    return config

class AdvancedColorAnalyzer:
    """Advanced color analysis using multiple color spaces and perceptual models"""
    
    def __init__(self):
        self.color_cache = {}
        self.kmeans_cache = {}
        
    @lru_cache(maxsize=1000)
    def rgb_to_lab(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Convert RGB to perceptually uniform LAB color space"""
        # Normalize RGB to [0,1]
        r, g, b = r/255.0, g/255.0, b/255.0
        
        # Apply gamma correction
        def gamma_correct(c):
            return c/12.92 if c <= 0.04045 else ((c + 0.055)/1.055) ** 2.4
        
        r, g, b = map(gamma_correct, [r, g, b])
        
        # Convert to XYZ
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505
        
        # Normalize by D65 illuminant
        x, y, z = x/0.95047, y/1.00000, z/1.08883
        
        # Convert to LAB
        def f(t):
            return t**(1/3) if t > 0.008856 else (7.787 * t + 16/116)
        
        fx, fy, fz = map(f, [x, y, z])
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return L, a, b
    
    def color_distance_lab(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate perceptual color distance in LAB space"""
        l1, a1, b1 = self.rgb_to_lab(*color1)
        l2, a2, b2 = self.rgb_to_lab(*color2)
        
        # CIE76 Delta E formula
        delta_e = ((l2 - l1)**2 + (a2 - a1)**2 + (b2 - b1)**2)**0.5
        return delta_e
    
    def extract_dominant_colors_advanced(self, image: Image.Image, n_colors: int = 5) -> List[Tuple[Tuple[int, int, int], float]]:
        """Extract dominant colors using advanced clustering with LAB color space"""
        cache_key = hash((id(image), n_colors))
        if cache_key in self.color_cache:
            return self.color_cache[cache_key]
        
        # Preprocess image for better color extraction
        img = image.convert('RGB')
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))  # Slight blur to reduce noise
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        pixels = np.array(img).reshape(-1, 3)
        
        # Remove very dark and very bright pixels (likely shadows/highlights)
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 30) & (brightness < 225)
        pixels = pixels[mask]
        
        if len(pixels) < 10:
            pixels = np.array(img).reshape(-1, 3)  # Fallback
        
        # Convert to LAB space for better clustering
        lab_pixels = np.array([self.rgb_to_lab(r, g, b) for r, g, b in pixels])
        
        # Use KMeans with better initialization
        kmeans = KMeans(
            n_clusters=min(n_colors, len(pixels)//10), 
            init='k-means++',
            n_init=10,
            random_state=42,
            max_iter=100
        )
        
        labels = kmeans.fit_predict(lab_pixels)
        
        # Convert cluster centers back to RGB
        colors_with_counts = []
        for i, center in enumerate(kmeans.cluster_centers_):
            # Convert LAB center back to RGB (approximate)
            count = np.sum(labels == i)
            fraction = count / len(pixels)
            
            # Find closest RGB pixel to LAB center
            distances = cdist([center], lab_pixels, metric='euclidean')[0]
            closest_idx = np.argmin(distances)
            rgb_color = tuple(pixels[closest_idx])
            
            colors_with_counts.append((rgb_color, fraction))
        
        # Sort by fraction (most dominant first)
        colors_with_counts.sort(key=lambda x: x[1], reverse=True)
        
        self.color_cache[cache_key] = colors_with_counts
        return colors_with_counts
    
    def get_color_harmony_score(self, color1: str, color2: str) -> float:
        """Advanced color harmony scoring based on color theory"""
        if not color1 or not color2:
            return 0.0
        
        # Exact match
        if color1 == color2:
            return 0.8
        
        # Neutral combinations
        if color1 in NEUTRALS or color2 in NEUTRALS:
            return 0.6
        
        # Complementary colors
        if color2 in COMPLEMENTARY_PAIRS.get(color1, []):
            return 0.9
        if color1 in COMPLEMENTARY_PAIRS.get(color2, []):
            return 0.9
        
        # Analogous colors (simplified)
        analogous_groups = [
            {"red", "orange", "pink"},
            {"blue", "teal", "cyan", "navy"},
            {"green", "lime", "olive"},
            {"yellow", "gold", "cream"},
            {"purple", "violet", "magenta"}
        ]
        
        for group in analogous_groups:
            if color1 in group and color2 in group:
                return 0.7
        
        # Default compatibility
        return 0.3

class OptimizedImagePreprocessor:
    """High-performance image preprocessing with caching and batch operations"""
    
    def __init__(self, img_size: int, config: PerformanceConfig):
        self.img_size = img_size
        self.config = config
        self.transform_cache = {}
        
        # Create optimized transforms
        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # GPU transforms if available
        if torch.cuda.is_available():
            self.gpu_transforms = transforms.Compose([
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Optimized image preprocessing with caching"""
        # Create cache key based on image content
        image_bytes = image.tobytes()
        cache_key = hashlib.md5(image_bytes + str(self.img_size).encode()).hexdigest()
        
        if cache_key in self.transform_cache and len(self.transform_cache) < self.config.cache_size:
            return self.transform_cache[cache_key]
        
        # Process image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transforms(image)
        
        # Cache result if cache not full
        if len(self.transform_cache) < self.config.cache_size:
            self.transform_cache[cache_key] = tensor
        
        return tensor
    
    def preprocess_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Batch preprocessing for better GPU utilization"""
        tensors = [self.preprocess_image(img) for img in images]
        return torch.stack(tensors)

class EnhancedCompatibilityModel(torch.nn.Module):
    """Enhanced compatibility model with attention and ensemble methods"""
    
    def __init__(self, backbone="efficientnet_b0", embed_dim=512, use_attention=True):
        super().__init__()
        import timm
        
        self.embed_dim = embed_dim
        self.use_attention = use_attention
        
        # Main encoder with dropout for regularization
        self.encoder = timm.create_model(
            backbone, 
            pretrained=False, 
            num_classes=embed_dim,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
        
        # Enhanced head with attention mechanism
        if use_attention:
            self.attention = torch.nn.MultiheadAttention(
                embed_dim, num_heads=8, dropout=0.1, batch_first=True
            )
            head_input_dim = embed_dim * 3  # concat + attention output
        else:
            head_input_dim = embed_dim * 2
        
        self.head = torch.nn.Sequential(
            torch.nn.Linear(head_input_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
        )
        
        # Ensemble prediction heads
        self.style_head = torch.nn.Linear(embed_dim * 2, 1)
        self.color_head = torch.nn.Linear(embed_dim * 2, 1)
        
    def forward(self, xa, xb):
        fa = self.encoder(xa)  # [B, D]
        fb = self.encoder(xb)  # [B, D]
        
        # Basic concatenation
        concat_features = torch.cat([fa, fb], dim=1)
        
        if self.use_attention:
            # Cross-attention between features
            features_stack = torch.stack([fa, fb], dim=1)  # [B, 2, D]
            attn_out, _ = self.attention(features_stack, features_stack, features_stack)
            attn_features = attn_out.mean(dim=1)  # [B, D]
            
            # Combine concat and attention features
            combined_features = torch.cat([concat_features, attn_features], dim=1)
        else:
            combined_features = concat_features
        
        # Main compatibility score
        main_score = self.head(combined_features).squeeze(1)
        
        # Ensemble scores
        style_score = self.style_head(concat_features).squeeze(1)
        color_score = self.color_head(concat_features).squeeze(1)
        
        # Weighted ensemble
        final_score = 0.6 * main_score + 0.25 * style_score + 0.15 * color_score
        
        return final_score

class HighPerformanceCatalogDataset(Dataset):
    """High-performance dataset with advanced caching and prefetching"""
    
    def __init__(self, items_df: pd.DataFrame, images_root: Path, preprocessor: OptimizedImagePreprocessor):
        self.df = items_df.reset_index(drop=True)
        self.root = Path(images_root)
        self.preprocessor = preprocessor
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Pre-validate image paths
        self.valid_indices = []
        for i, row in self.df.iterrows():
            image_path = self.root / row["image_path"]
            if image_path.exists():
                self.valid_indices.append(i)
            else:
                logger.warning(f"Image not found: {image_path}")
        
        logger.info(f"Valid images: {len(self.valid_indices)}/{len(self.df)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        item_id = str(row["item_id"])
        image_path = str(row["image_path"])
        
        # Check cache first
        if item_id in self.cache:
            self.cache_hits += 1
            tensor = self.cache[item_id]
        else:
            self.cache_misses += 1
            try:
                full_path = self.root / image_path
                with Image.open(full_path) as img:
                    img = img.convert("RGB")
                    tensor = self.preprocessor.preprocess_image(img)
                
                # Cache if space available
                if len(self.cache) < self.preprocessor.config.cache_size:
                    self.cache[item_id] = tensor
                    
            except Exception as e:
                logger.warning(f"Failed to load {image_path}: {e}")
                tensor = torch.zeros(3, self.preprocessor.img_size, self.preprocessor.img_size)
        
        return (
            item_id,
            image_path,
            str(row.get("category", "")),
            str(row.get("title", "")),
            tensor
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }

class AdvancedRecommendationEngine:
    """Advanced recommendation engine with enterprise-grade optimizations"""
    
    def __init__(
        self,
        ckpt_path: Path,
        catalog_csv: Path,
        images_root: Path,
        items_aug_csv: Optional[Path] = None,
        img_size: int = 224,
        device: Optional[torch.device] = None,
        item_cls_onnx: Optional[Path] = None,
        item_cls_meta: Optional[Path] = None,
        config: Optional[PerformanceConfig] = None
    ):
        # Initialize configuration
        self.config = config or get_optimal_config()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        
        # Initialize components
        self.color_analyzer = AdvancedColorAnalyzer()
        self.preprocessor = OptimizedImagePreprocessor(img_size, self.config)
        
        # Performance monitoring
        self.performance_stats = {
            "model_load_time": 0,
            "feature_computation_time": 0,
            "recommendation_time": 0,
            "cache_stats": {}
        }
        
        logger.info(f"Initializing AdvancedRecommendationEngine on {self.device}")
        logger.info(f"Configuration: {self.config}")
        
        # Load components
        self._load_model(ckpt_path)
        self._load_catalog_data(catalog_csv, images_root, items_aug_csv)
        self._load_item_classifier(item_cls_onnx, item_cls_meta)
        self._optimize_model()
        
        logger.info("AdvancedRecommendationEngine initialization complete")
    
    def _load_model(self, ckpt_path: Path):
        """Load model with enhanced error handling and validation"""
        import time
        start_time = time.time()
        
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
            
            backbone = meta.get("arch", "efficientnet_b0")
            embed_dim = meta.get("embed_dim", 512)
            
            logger.info(f"Loading model: backbone={backbone}, embed_dim={embed_dim}")
            
            # Create enhanced model
            self.model = EnhancedCompatibilityModel(
                backbone=backbone, 
                embed_dim=embed_dim,
                use_attention=True
            )
            
            # Load state dict with better error handling
            state_dict = self._extract_state_dict(checkpoint)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            # Validate model integrity
            self._validate_model_integrity(missing_keys, unexpected_keys)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.performance_stats["model_load_time"] = time.time() - start_time
            logger.info(f"Model loaded in {self.performance_stats['model_load_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _extract_state_dict(self, checkpoint_obj):
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
                
            # Handle backbone mapping for enhanced model
            if new_key.startswith("encoder.backbone."):
                new_key = new_key.replace("encoder.backbone.", "encoder.")
            elif not any(new_key.startswith(prefix) for prefix in ["encoder.", "head.", "attention.", "style_head.", "color_head."]):
                new_key = f"encoder.{new_key}"
                
            new_sd[new_key] = value

        return new_sd
    
    def _validate_model_integrity(self, missing_keys: List[str], unexpected_keys: List[str]):
        """Validate model loading and determine capabilities"""
        # Check if compatibility head is properly loaded
        head_missing = [k for k in missing_keys if k.startswith("head.")]
        attention_missing = [k for k in missing_keys if k.startswith("attention.")]
        
        self.head_available = len(head_missing) == 0
        self.attention_available = len(attention_missing) == 0
        
        if missing_keys:
            logger.warning(f"Missing keys: {len(missing_keys)} (first 5: {missing_keys[:5]})")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {len(unexpected_keys)} (first 5: {unexpected_keys[:5]})")
        
        logger.info(f"Model capabilities: head={self.head_available}, attention={self.attention_available}")
    
    def _load_catalog_data(self, catalog_csv: Path, images_root: Path, items_aug_csv: Optional[Path]):
        """Load catalog data with enhanced validation and preprocessing"""
        logger.info(f"Loading catalog from {catalog_csv}")
        
        try:
            # Load main catalog with better error handling
            self.items = pd.read_csv(
                catalog_csv,
                dtype={"item_id": str, "image_path": str, "title": str, "category": str},
                keep_default_na=False,
                na_values=[]
            ).fillna("")
            
            # Ensure required columns
            required_columns = ["item_id", "image_path", "title", "category"]
            for col in required_columns:
                if col not in self.items.columns:
                    self.items[col] = ""
            
            # Enhanced bucket classification
            self.items["bucket"] = self.items["category"].apply(self._classify_bucket_enhanced)
            
            # Remove invalid entries
            initial_count = len(self.items)
            self.items = self.items[
                (self.items["item_id"] != "") & 
                (self.items["image_path"] != "")
            ].reset_index(drop=True)
            
            removed_count = initial_count - len(self.items)
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} invalid catalog entries")
            
            self.images_root = Path(images_root)
            
            # Load enhanced color data
            self.color_lookup = {}
            if items_aug_csv and Path(items_aug_csv).exists():
                self._load_enhanced_color_data(items_aug_csv)
            
            logger.info(f"Loaded {len(self.items)} catalog items")
            logger.info(f"Bucket distribution: {self.items['bucket'].value_counts().to_dict()}")
            
        except Exception as e:
            logger.error(f"Failed to load catalog data: {e}")
            raise
    
    def _classify_bucket_enhanced(self, category: str) -> str:
        """Enhanced bucket classification with better keyword matching"""
        if not category:
            return "other"
        
        cat_lower = category.lower().strip()
        
        # Define comprehensive keyword mappings
        bucket_keywords = {
            "footwear": {
                "sneaker", "boot", "heel", "flat", "sandal", "loafer", "shoe", 
                "pump", "oxford", "ballet", "cleat", "slipper", "moccasin"
            },
            "outerwear": {
                "jacket", "coat", "blazer", "cardigan", "hoodie", "outer", 
                "parka", "windbreaker", "vest", "poncho", "cape", "trench"
            },
            "dress": {
                "dress", "gown", "jumper", "sundress", "maxi", "mini", 
                "cocktail", "evening", "wedding", "formal dress"
            },
            "bottoms": {
                "skirt", "jeans", "trouser", "pant", "short", "legging", "chino",
                "cargo", "capri", "culottes", "palazzo", "jogger", "sweatpant"
            },
            "tops": {
                "top", "tee", "t-shirt", "shirt", "blouse", "sweater", "pullover", 
                "tank", "camisole", "tunic", "henley", "polo", "crop", "bodysuit"
            }
        }
        
        # Score each bucket based on keyword matches
        bucket_scores = {}
        for bucket, keywords in bucket_keywords.items():
            score = sum(1 for keyword in keywords if keyword in cat_lower)
            if score > 0:
                bucket_scores[bucket] = score
        
        # Return bucket with highest score, or "other" if no matches
        if bucket_scores:
            return max(bucket_scores, key=bucket_scores.get)
        
        return "other"
    
    def _load_enhanced_color_data(self, items_aug_csv: Path):
        """Load and enhance color data with better processing"""
        try:
            aug = pd.read_csv(items_aug_csv, dtype=str, keep_default_na=False)
            
            color_data = {}
            for _, row in aug.iterrows():
                image_path = str(row.get("image_path", ""))
                if image_path:
                    # Enhanced color extraction
                    primary_color = self._normalize_color_name(row.get("primary_name", ""))
                    secondary_color = self._normalize_color_name(row.get("secondary_name", ""))
                    
                    try:
                        primary_frac = float(row.get("primary_frac", "0.0"))
                        secondary_frac = float(row.get("secondary_frac", "0.0"))
                    except (ValueError, TypeError):
                        primary_frac, secondary_frac = 0.0, 0.0
                    
                    # Choose best color representation
                    if primary_color and primary_frac > 0.15:
                        color_data[image_path] = (primary_color, primary_frac)
                    elif secondary_color and secondary_frac > 0.15:
                        color_data[image_path] = (secondary_color, secondary_frac)
                    else:
                        color_data[image_path] = ("neutral", 1.0)
            
            self.color_lookup = color_data
            logger.info(f"Loaded enhanced color data for {len(self.color_lookup)} items")
            
        except Exception as e:
            logger.warning(f"Failed to load color data: {e}")
            self.color_lookup = {}
    
    def _normalize_color_name(self, name: str) -> str:
        """Enhanced color name normalization"""
        if not isinstance(name, str):
            return ""
        
        normalized = name.strip().lower()
        
        # Color name mappings for consistency
        color_mappings = {
            "grey": "gray",
            "navy blue": "navy",
            "light blue": "blue",
            "dark blue": "navy",
            "light green": "green",
            "dark green": "green",
            "hot pink": "pink",
            "light pink": "pink",
            "dark red": "red",
            "maroon": "red",
            "olive green": "olive",
            "lime green": "lime"
        }
        
        return color_mappings.get(normalized, normalized)
    
    def _load_item_classifier(self, item_cls_onnx: Optional[Path], item_cls_meta: Optional[Path]):
        """Load ONNX item classifier with enhanced error handling"""
        self.cls = None
        self.cls_labels = []
        self.cls_img_size = 224
        
        if not item_cls_onnx or not Path(item_cls_onnx).exists() or not ONNX_AVAILABLE:
            logger.info("ONNX classifier not available")
            return
        
        try:
            # Optimize providers based on hardware
            providers = ["CPUExecutionProvider"]
            if torch.cuda.is_available():
                providers.insert(0, "CUDAExecutionProvider")
            
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            
            self.cls = ort.InferenceSession(
                str(item_cls_onnx), 
                providers=providers,
                sess_options=session_options
            )
            
            # Load metadata with validation
            if item_cls_meta and Path(item_cls_meta).exists():
                with open(item_cls_meta, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    self.cls_labels = meta.get("class_names", [])
                    self.cls_img_size = int(meta.get("img_size", 224))
            
            # Enhanced label to bucket mapping
            self._label_to_bucket = {
                "jeans": "bottoms", "trouser": "bottoms", "trousers": "bottoms",
                "shorts": "bottoms", "leggings": "bottoms", "skirt": "bottoms",
                "pants": "bottoms", "chinos": "bottoms",
                "shirt": "tops", "blouse": "tops", "t-shirt": "tops", "tee": "tops",
                "sweater": "tops", "pullover": "tops", "tank": "tops", "top": "tops",
                "jacket": "outerwear", "coat": "outerwear", "blazer": "outerwear",
                "cardigan": "outerwear", "hoodie": "outerwear", "vest": "outerwear",
                "sneakers": "footwear", "boots": "footwear", "heels": "footwear",
                "sandals": "footwear", "shoes": "footwear", "flats": "footwear",
                "dress": "dress", "gown": "dress", "sundress": "dress"
            }
            
            logger.info(f"ONNX classifier loaded: {len(self.cls_labels)} classes, img_size={self.cls_img_size}")
            
        except Exception as e:
            logger.warning(f"Failed to load ONNX classifier: {e}")
            self.cls = None
    
    def _optimize_model(self):
        """Advanced model optimization with multiple techniques"""
        try:
            # Enable cuDNN benchmarking
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            # Warmup model
            self._warmup_model()
            
            # Advanced optimizations for modern GPUs
            if self.device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7:
                try:
                    # Enable channels-last memory format for better performance
                    self.model = self.model.to(memory_format=torch.channels_last)
                    
                    # Compile model with PyTorch 2.x (if available)
                    if self.config.compile_model and hasattr(torch, 'compile'):
                        self.model = torch.compile(
                            self.model, 
                            mode="reduce-overhead",
                            fullgraph=False
                        )
                        logger.info("Applied torch.compile optimization")
                    
                    logger.info("Applied channels_last memory format")
                    
                except Exception as e:
                    logger.warning(f"Advanced GPU optimizations failed: {e}")
            
            # Enable mixed precision if supported
            if self.config.mixed_precision:
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("Mixed precision enabled")
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
    
    def _warmup_model(self):
        """Comprehensive model warmup with multiple input sizes"""
        try:
            warmup_sizes = [1, 4] if self.config.batch_size >= 4 else [1]
            
            for batch_size in warmup_sizes:
                dummy_input = torch.randn(
                    batch_size, 3, self.img_size, self.img_size, 
                    device=self.device,
                    memory_format=torch.channels_last if self.device.type == "cuda" else torch.contiguous_format
                )
                
                with torch.no_grad():
                    if self.config.mixed_precision:
                        with torch.amp.autocast(device_type=self.device.type):
                            _ = self.model.encoder(dummy_input)
                    else:
                        _ = self.model.encoder(dummy_input)
            
            logger.info("Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    @torch.no_grad()
    def predict_item_category_advanced(self, image: Image.Image, confidence_threshold: float = 0.3) -> Tuple[str, float]:
        """Advanced item category prediction with ensemble methods"""
        if self.cls is None:
            return self._predict_category_heuristic(image)
        
        try:
            # Preprocess image for classifier
            img_resized = image.convert("RGB").resize(
                (self.cls_img_size, self.cls_img_size), 
                Image.Resampling.BICUBIC
            )
            
            arr = np.asarray(img_resized, dtype=np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            arr = np.transpose(arr, (2, 0, 1))[None, ...]
            
            # Run inference
            input_name = self.cls.get_inputs()[0].name
            output_name = self.cls.get_outputs()[0].name
            logits = self.cls.run([output_name], {input_name: arr})[0]
            
            # Get predictions
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            
            if confidence >= confidence_threshold and pred_idx < len(self.cls_labels):
                label = self.cls_labels[pred_idx]
                bucket = self._label_to_bucket.get(label.lower(), "other")
                
                logger.debug(f"Classified: {label} -> {bucket} (conf: {confidence:.3f})")
                return bucket, confidence
            else:
                # Fall back to heuristic method
                return self._predict_category_heuristic(image)
                
        except Exception as e:
            logger.debug(f"Classifier prediction failed: {e}")
            return self._predict_category_heuristic(image)
    
    def _predict_category_heuristic(self, image: Image.Image) -> Tuple[str, float]:
        """Improved heuristic category prediction"""
        try:
            width, height = image.size
            aspect_ratio = width / height
            
            # Enhanced heuristics based on fashion domain knowledge
            if aspect_ratio > 2.0:  # Very wide items - likely tops laid flat
                return "tops", 0.6
            elif aspect_ratio < 0.4:  # Very tall items - likely dresses or long outerwear
                return "outerwear", 0.5
            elif 0.7 <= aspect_ratio <= 1.3:  # Square-ish items
                # Use image analysis for better classification
                dominant_colors = self.color_analyzer.extract_dominant_colors_advanced(image, n_colors=3)
                
                # If very dark (likely formal wear)
                if dominant_colors and self._is_dark_dominant(dominant_colors):
                    return "outerwear", 0.4
                else:
                    return "tops", 0.5
            else:
                return "bottoms", 0.4
                
        except Exception as e:
            logger.debug(f"Heuristic prediction failed: {e}")
            return "other", 0.1
    
    def _is_dark_dominant(self, color_data: List[Tuple[Tuple[int, int, int], float]]) -> bool:
        """Check if dark colors dominate the image"""
        dark_fraction = 0.0
        for (r, g, b), fraction in color_data:
            brightness = (r + g + b) / 3
            if brightness < 80:  # Dark threshold
                dark_fraction += fraction
        
        return dark_fraction > 0.5
    
    @torch.no_grad()
    def recommend_advanced(
        self,
        user_image: Image.Image,
        *,
        allow_types: str = "tops,bottoms,outerwear,footwear",
        per_bucket: int = 5,
        topk: int = 20,
        color_weight: float = 0.15,
        style_weight: float = 0.1,
        diversity_weight: float = 0.05,
        anchor_color_mode: str = "advanced",
        filter_same_bucket: bool = True,
        use_ensemble: bool = True
    ) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
        """Advanced recommendation with comprehensive scoring and analytics"""
        
        import time
        start_time = time.time()
        
        # Initialize analytics
        analytics = {
            "processing_time": 0,
            "anchor_detection": {},
            "filtering_stats": {},
            "scoring_stats": {},
            "diversity_stats": {}
        }
        
        # Set seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Advanced anchor detection
        anchor_color, anchor_bucket, color_confidence = self._detect_anchor_advanced(
            user_image, anchor_color_mode
        )
        
        analytics["anchor_detection"] = {
            "color": anchor_color,
            "bucket": anchor_bucket,
            "color_confidence": color_confidence,
            "mode": anchor_color_mode
        }
        
        logger.info(f"Detected anchor - color: {anchor_color}, bucket: {anchor_bucket}")
        
        # Advanced catalog filtering
        filtered_items = self._filter_catalog_advanced(
            allow_types, anchor_bucket, filter_same_bucket
        )
        
        analytics["filtering_stats"] = {
            "original_count": len(self.items),
            "filtered_count": len(filtered_items),
            "filter_ratio": len(filtered_items) / max(len(self.items), 1)
        }
        
        if len(filtered_items) == 0:
            logger.warning("No items found after filtering")
            return anchor_color, pd.DataFrame(), analytics
        
        # Encode user image with optimizations
        user_tensor = self.preprocessor.preprocess_image(user_image).unsqueeze(0).to(self.device)
        
        with torch.amp.autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
            user_features = self.model.encoder(user_tensor)
        
        # Advanced scoring with ensemble methods
        if use_ensemble:
            scores, score_components = self._score_with_ensemble(
                filtered_items, user_features, anchor_color, 
                color_weight, style_weight
            )
        else:
            scores, score_components = self._score_basic(
                filtered_items, user_features, anchor_color, color_weight
            )
        
        analytics["scoring_stats"] = score_components
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'item_id': filtered_items['item_id'].values,
            'image_path': filtered_items['image_path'].values,
            'category': filtered_items['category'].values,
            'title': filtered_items['title'].values,
            'bucket': filtered_items['bucket'].values,
            'score': scores
        })
        
        # Advanced diversification with analytics
        final_results, diversity_stats = self._diversify_advanced(
            results_df, per_bucket, topk, diversity_weight
        )
        
        analytics["diversity_stats"] = diversity_stats
        analytics["processing_time"] = time.time() - start_time
        
        logger.info(f"Recommendation completed in {analytics['processing_time']:.2f}s")
        
        return anchor_color, final_results, analytics
    
    def _detect_anchor_advanced(self, image: Image.Image, mode: str) -> Tuple[str, str, float]:
        """Advanced anchor detection with confidence scoring"""
        
        # Detect category with confidence
        anchor_bucket, category_confidence = self.predict_item_category_advanced(image)
        
        # Advanced color detection
        if mode == "advanced":
            # Use multiple methods and ensemble
            dominant_colors = self.color_analyzer.extract_dominant_colors_advanced(image, n_colors=5)
            
            if dominant_colors:
                # Choose color based on fashion significance
                anchor_color = self._select_fashion_significant_color(dominant_colors)
                color_confidence = dominant_colors[0][1]  # Use fraction of most dominant
            else:
                anchor_color = "neutral"
                color_confidence = 0.5
        
        elif mode == "kmeans":
            anchor_color = self._dominant_color_kmeans_advanced(image)
            color_confidence = 0.7
        
        else:  # hsv mode
            anchor_color = self._dominant_color_hsv_advanced(image)
            color_confidence = 0.6
        
        return anchor_color, anchor_bucket, color_confidence
    
    def _select_fashion_significant_color(self, color_data: List[Tuple[Tuple[int, int, int], float]]) -> str:
        """Select the most fashion-significant color from dominant colors"""
        
        # Convert RGB colors to color names
        color_names = []
        for (r, g, b), fraction in color_data:
            l, a, b_lab = self.color_analyzer.rgb_to_lab(r, g, b)
            color_name = self._rgb_to_color_name_advanced(r, g, b)
            color_names.append((color_name, fraction))
        
        # Priority order for fashion significance
        fashion_priority = [
            "red", "blue", "green", "purple", "orange", "yellow", "pink",
            "navy", "teal", "coral", "burgundy", "forest", "olive",
            "black", "white", "gray", "brown", "beige", "cream"
        ]
        
        # Select highest priority color with reasonable fraction
        for priority_color in fashion_priority:
            for color_name, fraction in color_names:
                if color_name == priority_color and fraction > 0.1:
                    return color_name
        
        # Fallback to most dominant
        return color_names[0][0] if color_names else "neutral"
    
    def _rgb_to_color_name_advanced(self, r: int, g: int, b: int) -> str:
        """Advanced RGB to color name mapping with better accuracy"""
        
        # Convert to HSV for better color classification
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        h *= 360  # Convert to degrees
        
        # Advanced color classification
        if v < 0.15:
            return "black"
        elif s < 0.1:
            return "white" if v > 0.85 else "gray"
        elif s < 0.3 and v > 0.7:
            return "cream" if 30 <= h <= 60 else "gray"
        
        # Hue-based classification with better boundaries
        if h < 10 or h >= 350:
            return "red"
        elif h < 25:
            return "coral"
        elif h < 45:
            return "orange"
        elif h < 65:
            return "yellow"
        elif h < 80:
            return "lime"
        elif h < 140:
            return "green"
        elif h < 170:
            return "teal"
        elif h < 200:
            return "cyan"
        elif h < 225:
            return "blue"
        elif h < 245:
            return "navy"
        elif h < 280:
            return "purple"
        elif h < 320:
            return "magenta"
        elif h < 350:
            return "pink"
        
        return "other"
    
    def _filter_catalog_advanced(self, allow_types: str, anchor_bucket: str, filter_same_bucket: bool) -> pd.DataFrame:
        """Advanced catalog filtering with intelligent logic"""
        
        filtered = self.items.copy()
        
        # Apply allowed types filter
        if allow_types:
            allowed = set(t.strip().lower() for t in allow_types.split(",") if t.strip())
            if allowed:
                filtered = filtered[filtered["bucket"].isin(allowed)]
        
        # Advanced same-bucket filtering logic
        if filter_same_bucket and anchor_bucket in {"tops", "bottoms", "outerwear", "footwear", "dress"}:
            # For dresses, don't filter out other dresses (they can be layered)
            if anchor_bucket == "dress":
                # Only filter if we have many dress items
                dress_count = (filtered["bucket"] == "dress").sum()
                if dress_count > 10:
                    filtered = filtered[filtered["bucket"] != "dress"]
            else:
                filtered = filtered[filtered["bucket"] != anchor_bucket]
        
        # Ensure minimum diversity
        bucket_counts = filtered["bucket"].value_counts()
        if len(bucket_counts) == 1 and len(filtered) > 50:
            # If only one bucket remains, bring back some variety
            other_buckets = self.items[~self.items["bucket"].isin(filtered["bucket"].unique())]
            if len(other_buckets) > 0:
                # Add top 20% from other buckets
                sample_size = min(len(other_buckets), len(filtered) // 5)
                additional_items = other_buckets.sample(n=sample_size, random_state=42)
                filtered = pd.concat([filtered, additional_items], ignore_index=True)
        
        return filtered.reset_index(drop=True)
    
    def _score_with_ensemble(
        self, 
        items: pd.DataFrame, 
        user_features: torch.Tensor, 
        anchor_color: str,
        color_weight: float,
        style_weight: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Advanced ensemble scoring with multiple components"""
        
        if len(items) == 0:
            return np.array([]), {}
        
        # Create high-performance dataset
        dataset = HighPerformanceCatalogDataset(items, self.images_root, self.preprocessor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=(self.config.num_workers > 0 and len(items) > 100),
        )
        
        # Score components
        compatibility_scores = []
        color_scores = []
        style_scores = []
        
        with torch.amp.autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
            for batch_data in dataloader:
                item_ids, image_paths, categories, titles, item_tensors = batch_data
                item_tensors = item_tensors.to(self.device, non_blocking=True)
                
                # Extract features
                item_features = self.model.encoder(item_tensors)
                batch_size = item_features.shape[0]
                
                # Main compatibility scores
                if self.head_available:
                    user_repeated = user_features.repeat(batch_size, 1)
                    
                    # Use ensemble prediction from model
                    if hasattr(self.model, 'style_head'):
                        combined_features = torch.cat([user_repeated, item_features], dim=1)
                        main_scores = torch.sigmoid(self.model.head(combined_features)).squeeze(1)
                        style_batch_scores = torch.sigmoid(self.model.style_head(combined_features)).squeeze(1)
                        color_batch_scores = torch.sigmoid(self.model.color_head(combined_features)).squeeze(1)
                    else:
                        # Fallback to basic head
                        combined_features = torch.cat([user_repeated, item_features], dim=1)
                        main_scores = torch.sigmoid(self.model.head(combined_features)).squeeze(1)
                        style_batch_scores = main_scores.clone()
                        color_batch_scores = main_scores.clone()
                else:
                    # Cosine similarity fallback
                    main_scores = F.cosine_similarity(
                        user_features.repeat(batch_size, 1), item_features, dim=1
                    )
                    main_scores = (main_scores + 1.0) * 0.5
                    style_batch_scores = main_scores.clone()
                    color_batch_scores = main_scores.clone()
                
                compatibility_scores.extend(main_scores.detach().cpu().numpy())
                style_scores.extend(style_batch_scores.detach().cpu().numpy())
                color_scores.extend(color_batch_scores.detach().cpu().numpy())
        
        # Convert to numpy arrays
        compatibility_scores = np.array(compatibility_scores)
        style_scores = np.array(style_scores)
        color_scores = np.array(color_scores)
        
        # Apply color harmony bonuses
        color_bonuses = self._compute_color_bonuses(items, anchor_color)
        
        # Combine scores with weights
        final_scores = (
            0.6 * compatibility_scores +
            0.15 * style_scores +
            0.1 * color_scores +
            color_weight * color_bonuses
        )
        
        # Normalize to [0, 1]
        final_scores = np.clip(final_scores, 0, 1)
        
        score_components = {
            "compatibility_mean": float(np.mean(compatibility_scores)),
            "style_mean": float(np.mean(style_scores)),
            "color_mean": float(np.mean(color_scores)),
            "color_bonus_mean": float(np.mean(color_bonuses)),
            "final_mean": float(np.mean(final_scores)),
            "cache_stats": dataset.get_cache_stats()
        }
        
        return final_scores, score_components
    
    def _compute_color_bonuses(self, items: pd.DataFrame, anchor_color: str) -> np.ndarray:
        """Compute color harmony bonuses for items"""
        bonuses = np.zeros(len(items))
        
        if not anchor_color or not self.color_lookup:
            return bonuses
        
        for i, (_, row) in enumerate(items.iterrows()):
            img_path = str(row["image_path"])
            item_color, frac = self.color_lookup.get(img_path, ("", 1.0))
            
            if item_color:
                harmony_score = self.color_analyzer.get_color_harmony_score(anchor_color, item_color)
                bonuses[i] = harmony_score * max(0.5, float(frac or 1.0))
        
        return bonuses
    
    def _diversify_advanced(
        self, 
        results_df: pd.DataFrame, 
        per_bucket: int, 
        topk: int, 
        diversity_weight: float
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Advanced diversification with analytics"""
        
        if len(results_df) == 0:
            return pd.DataFrame(), {}
        
        diversified_pieces = []
        bucket_stats = {}
        
        # Group by bucket and select diverse items
        for bucket, group in results_df.groupby("bucket"):
            # Sort by score
            sorted_group = group.sort_values("score", ascending=False)
            
            # Apply diversity within bucket
            if diversity_weight > 0 and len(sorted_group) > per_bucket:
                diverse_selection = self._apply_diversity_sampling(
                    sorted_group, per_bucket, diversity_weight
                )
            else:
                diverse_selection = sorted_group.head(per_bucket)
            
            diversified_pieces.append(diverse_selection)
            bucket_stats[bucket] = {
                "total_items": len(group),
                "selected_items": len(diverse_selection),
                "avg_score": float(diverse_selection["score"].mean()),
                "score_range": [float(diverse_selection["score"].min()), float(diverse_selection["score"].max())]
            }
        
        # Combine and final ranking
        if diversified_pieces:
            final_df = pd.concat(diversified_pieces, ignore_index=True)
            final_df = final_df.sort_values("score", ascending=False).head(topk)
            
            diversity_stats = {
                "bucket_distribution": bucket_stats,
                "final_count": len(final_df),
                "buckets_represented": len(final_df["bucket"].unique()),
                "score_distribution": {
                    "mean": float(final_df["score"].mean()),
                    "std": float(final_df["score"].std()),
                    "min": float(final_df["score"].min()),
                    "max": float(final_df["score"].max())
                }
            }
            
            return final_df.reset_index(drop=True), diversity_stats
        
        return pd.DataFrame(), {}
    
    def _apply_diversity_sampling(self, group: pd.DataFrame, target_count: int, diversity_weight: float) -> pd.DataFrame:
        """Apply diversity sampling within a group"""
        
        if len(group) <= target_count:
            return group
        
        # Start with top item
        selected_indices = [0]
        remaining_indices = list(range(1, len(group)))
        
        # Iteratively select diverse items
        while len(selected_indices) < target_count and remaining_indices:
            best_idx = None
            best_score = -1
            
            for idx in remaining_indices:
                # Score combines ranking position and diversity
                position_score = (len(group) - idx) / len(group)  # Higher for better ranked items
                
                # Simple diversity score based on category differences
                diversity_score = self._compute_diversity_score(
                    group.iloc[idx], [group.iloc[i] for i in selected_indices]
                )
                
                combined_score = (1 - diversity_weight) * position_score + diversity_weight * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return group.iloc[selected_indices]
    
    def _compute_diversity_score(self, candidate_item: pd.Series, selected_items: List[pd.Series]) -> float:
        """Compute diversity score for an item against selected items"""
        
        if not selected_items:
            return 1.0
        
        # Simple diversity based on title/category differences
        candidate_title = str(candidate_item.get("title", "")).lower()
        candidate_category = str(candidate_item.get("category", "")).lower()
        
        diversity_scores = []
        
        for selected_item in selected_items:
            selected_title = str(selected_item.get("title", "")).lower()
            selected_category = str(selected_item.get("category", "")).lower()
            
            # Simple text similarity (inverse)
            title_similarity = len(set(candidate_title.split()) & set(selected_title.split())) / max(
                len(set(candidate_title.split()) | set(selected_title.split())), 1
            )
            
            category_similarity = 1.0 if candidate_category == selected_category else 0.0
            
            # Diversity is inverse of similarity
            item_diversity = 1.0 - (0.7 * title_similarity + 0.3 * category_similarity)
            diversity_scores.append(item_diversity)
        
        return np.mean(diversity_scores)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.performance_stats.copy()
        
        # Add current system stats
        if torch.cuda.is_available():
            stats["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated() / 1e9,
                "reserved": torch.cuda.memory_reserved() / 1e9,
                "max_allocated": torch.cuda.max_memory_allocated() / 1e9
            }
        
        # Model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        stats["model_info"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / 1e6,  # Assuming float32
            "device": str(self.device),
            "mixed_precision": self.config.mixed_precision
        }
        
        return stats
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'preprocessor') and hasattr(self.preprocessor, 'transform_cache'):
                self.preprocessor.transform_cache.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception:
            pass  # Ignore cleanup errors