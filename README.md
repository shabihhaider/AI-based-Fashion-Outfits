# AI-based Fashion Outfits

## Project Goals

Build a deep learning system that recommends visually compatible fashion outfits. The project covers dataset preparation, model training, and an API for serving outfit recommendations.

## Data Sources

- **DeepFashion MultiModal** – item attributes and images.
- **Polyvore Outfits** – outfit compositions and item images for learning compatibility.

## Features

- Scripts for preprocessing raw datasets into a unified format.
- Training pipeline for an image-only outfit compatibility model.
- FastAPI service and demo scripts for generating outfit recommendations.

## Installation

Using pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
```

Using conda:
```bash
conda env create -f env/conda_env.yaml
conda activate outfit
```

## Dataset Setup & Configuration

Place raw datasets at:
```
data/raw/deepfashion_mm/
data/raw/polyvore_outfit/
```
Preprocessed outputs are written under `data/processed/...`. Adjust `configs/paths.yaml` if your data lives elsewhere.

## Preprocessing

Create processed manifests and images:
```bash
python scripts/prepare_polyvore.py --variant disjoint
python scripts/prepare_deepfashion_mm.py
```

## Training

Train the compatibility model:
```bash
python scripts/train_compatibility.py --epochs 6 --batch-size 32
```

## Running the API

Launch the FastAPI service:
```bash
python app/main.py
# or: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For an offline recommendation demo:
```bash
python scripts/recommend_api.py \
    --ckpt outputs/models/compatibility_best.pt \
    --catalog-csv data/processed/polyvore/manifests/items_test.csv \
    --images-root data/processed/polyvore/images \
    --user-image path/to/photo.jpg
```
