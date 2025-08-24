#!/usr/bin/env python3
"""
Improved cache generation script with better error handling and validation
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_image_preproc(img: Image.Image, img_size: int) -> torch.Tensor:
    """Preprocess image for model input"""
    im = img.convert("RGB").resize((img_size, img_size), Image.Resampling.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

class CatalogDS(Dataset):
    """Dataset for catalog items with error handling"""
    def __init__(self, items_csv: Path, images_root: Path, img_size: int):
        logger.info(f"Loading catalog from {items_csv}")
        self.df = pd.read_csv(
            items_csv, 
            dtype={"item_id": str, "image_path": str, "category": str, "title": str},
            keep_default_na=False
        ).fillna("")
        
        self.root = Path(images_root)
        self.img_size = img_size
        
        # Validate that image files exist
        missing_count = 0
        valid_indices = []
        
        for i, row in self.df.iterrows():
            image_path = self.root / row["image_path"]
            if image_path.exists():
                valid_indices.append(i)
            else:
                missing_count += 1
                if missing_count <= 10:  # Log first 10 missing files
                    logger.warning(f"Missing image: {image_path}")
        
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing images out of {len(self.df)} total")
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)
            logger.info(f"Using {len(self.df)} valid items")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            img_path = self.root / r["image_path"]
            x = Image.open(img_path)
            x = load_image_preproc(x, self.img_size)
            return r["image_path"], x
        except Exception as e:
            logger.error(f"Failed to load image {r['image_path']}: {e}")
            # Return a dummy tensor
            return r["image_path"], torch.zeros(3, self.img_size, self.img_size)

def build_encoder(backbone="efficientnet_b0", embed_dim=512):
    """Build encoder model"""
    import timm
    import torch.nn as nn
    
    model = timm.create_model(backbone, pretrained=False, num_classes=embed_dim)
    
    class EncWrap(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, x):
            return self.enc(x)
    
    return EncWrap(model)

def extract_state_dict(checkpoint_obj):
    """Extract state dict with enhanced handling"""
    if isinstance(checkpoint_obj, dict):
        # Try multiple possible keys
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                sd = checkpoint_obj[key]
                break
        else:
            # Check if dict itself looks like state dict
            if any(isinstance(v, torch.Tensor) for v in checkpoint_obj.values()):
                sd = checkpoint_obj
            else:
                raise RuntimeError("Could not locate state_dict in checkpoint")
    else:
        sd = checkpoint_obj

    if sd is None:
        raise RuntimeError("Could not extract state_dict from checkpoint")

    # Clean up keys and extract encoder weights
    new_sd = {}
    for key, value in sd.items():
        new_key = key
        
        # Remove common prefixes
        if new_key.startswith("module."):
            new_key = new_key[7:]
        if new_key.startswith("model."):
            new_key = new_key[6:]
        
        # Keep only encoder weights, remove encoder. prefix
        if new_key.startswith("encoder."):
            encoder_key = new_key[8:]  # Remove "encoder." prefix
            new_sd[encoder_key] = value
        elif not ("head." in new_key or "classifier" in new_key):
            # If it's not a head/classifier weight and doesn't have encoder prefix,
            # assume it belongs to encoder
            new_sd[new_key] = value
    
    return new_sd

def main():
    parser = argparse.ArgumentParser(description="Generate catalog feature cache")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--catalog-csv", required=True, help="Path to catalog CSV")
    parser.add_argument("--images-root", required=True, help="Path to images root directory")
    parser.add_argument("--out", required=True, help="Output path for feature cache")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cuda/cpu)")
    args = parser.parse_args()

    # Validate inputs
    ckpt_path = Path(args.ckpt)
    catalog_path = Path(args.catalog_csv)
    images_root = Path(args.images_root)
    output_path = Path(args.out)
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog CSV not found: {catalog_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load checkpoint and extract metadata
    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    backbone = meta.get("arch", "efficientnet_b0")
    embed_dim = int(meta.get("embed_dim", 512))
    
    logger.info(f"Model config - backbone: {backbone}, embed_dim: {embed_dim}")
    
    # Build encoder
    encoder = build_encoder(backbone, embed_dim).to(device)
    
    # Load state dict
    try:
        enc_sd = extract_state_dict(checkpoint)
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
        if missing:
            logger.warning(f"Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
        logger.info("Encoder loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load encoder weights: {e}")
        raise
    
    # Create dataset and dataloader
    logger.info("Creating dataset...")
    dataset = CatalogDS(catalog_path, images_root, args.img_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0)
    )
    
    logger.info(f"Processing {len(dataset)} items in batches of {args.batch_size}")
    
    # Extract features
    features = torch.empty((len(dataset), embed_dim), dtype=torch.float32)
    paths = []
    idx = 0
    
    encoder.eval()
    with torch.no_grad():
        for batch_idx, (batch_paths, batch_images) in enumerate(dataloader):
            batch_size = batch_images.size(0)
            batch_images = batch_images.to(device)
            
            try:
                # Extract features
                batch_features = encoder(batch_images).detach().float().cpu()
                features[idx:idx + batch_size] = batch_features
                paths.extend(batch_paths)
                idx += batch_size
                
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx}/{len(dataset)} items ({100 * idx / len(dataset):.1f}%)")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Fill with zeros and continue
                features[idx:idx + batch_size] = torch.zeros(batch_size, embed_dim)
                paths.extend(batch_paths)
                idx += batch_size
    
    logger.info(f"Feature extraction complete: {features.shape}")
    
    # Save cache
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "feats": features,
        "paths": paths,
        "img_size": args.img_size,
        "backbone": backbone,
        "embed_dim": embed_dim,
        "version": "2.0",
        "num_items": len(dataset)
    }
    
    logger.info(f"Saving cache to {output_path}")
    torch.save(cache_data, output_path)
    
    # Verify cache
    logger.info("Verifying saved cache...")
    try:
        loaded = torch.load(output_path, map_location="cpu")
        assert loaded["feats"].shape == (len(dataset), embed_dim)
        assert len(loaded["paths"]) == len(dataset)
        assert loaded["img_size"] == args.img_size
        assert loaded["embed_dim"] == embed_dim
        logger.info("Cache verification successful")
    except Exception as e:
        logger.error(f"Cache verification failed: {e}")
        raise
    
    logger.info(f"✅ Successfully created feature cache: {output_path}")
    logger.info(f"   - Features: {features.shape}")
    logger.info(f"   - Items: {len(paths)}")
    logger.info(f"   - Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()