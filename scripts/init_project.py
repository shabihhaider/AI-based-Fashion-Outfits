from pathlib import Path
import textwrap, json, yaml

# === Change this if you want a different root ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIRS = [
    "configs",
    "data/raw/deepfashion_mm",
    "data/raw/polyvore_outfit",
    "data/processed/deepfashion_mm/images",
    "data/processed/deepfashion_mm/manifests",
    "data/processed/polyvore/images",
    "data/processed/polyvore/manifests",
    "scripts",
    "env",
    "notebooks",
    "models",
    "outputs/logs",
    "outputs/checkpoints",
]

labels_yaml = {
    "version": 1,
    "note": "Clothes only (no bags/jewelry). Extend as needed.",
    "categories": {
        "tops": ["t_shirt", "shirt", "blouse", "sweater", "hoodie", "tank_top", "polo", "cardigan"],
        "bottoms": ["jeans", "trousers", "shorts", "skirt", "leggings", "chinos"],
        "outerwear": ["jacket", "coat", "blazer", "raincoat", "parka", "windbreaker"],
        "dresses": ["dress_mini", "dress_knee", "dress_midi", "dress_maxi", "jumpsuit"],
        # footwear intentionally excluded for now
    },
    "attributes": {
        "color_primary": ["black","white","gray","red","orange","yellow","green","blue","purple","pink","brown","beige","navy","olive","cyan"],
        "color_secondary": ["none","black","white","gray","red","orange","yellow","green","blue","purple","pink","brown","beige","navy","olive","cyan"],
        "pattern": ["solid","striped","plaid","floral","polka_dot","geometric","abstract","graphic","animal_print"],
        "fabric": ["cotton","denim","wool","silk","linen","leather","polyester","nylon","rayon","knit","fleece"],
        "style": ["casual","business_casual","formal","sporty","streetwear","vintage","bohemian","minimalist"],
        "fit": ["tight","regular","loose","oversized"],
        "sleeve_length": ["sleeveless","short","three_quarter","long"],
        "neckline": ["crew","v_neck","scoop","turtleneck","collared","off_shoulder"],
        "length": ["cropped","regular","long","mini","knee","midi","maxi"],
        "season": ["spring","summer","fall","winter"]
    }
}

paths_yaml = {
    "root": str(PROJECT_ROOT),
    "raw": {
        "deepfashion_mm": str(PROJECT_ROOT / "data/raw/deepfashion_mm"),
        "polyvore_outfit": str(PROJECT_ROOT / "data/raw/polyvore_outfit"),
    },
    "processed": {
        "deepfashion_mm": {
            "images": str(PROJECT_ROOT / "data/processed/deepfashion_mm/images"),
            "manifests": str(PROJECT_ROOT / "data/processed/deepfashion_mm/manifests"),
        },
        "polyvore": {
            "images": str(PROJECT_ROOT / "data/processed/polyvore/images"),
            "manifests": str(PROJECT_ROOT / "data/processed/polyvore/manifests"),
        }
    }
}

requirements = """\
# minimal preprocessing + manifests
numpy
pandas
pillow
opencv-python
tqdm
pyyaml
scikit-learn
"""

conda_env = {
    "name": "outfit-prep",
    "channels": ["conda-forge", "defaults"],
    "dependencies": [
        "python=3.10",
        "pip",
        {"pip": [r.strip() for r in requirements.splitlines() if r and not r.startswith("#")]}
    ],
}

readme = """\
# Outfit Recommender — Data & Training

- Put original datasets in:
  - `data/raw/deepfashion_mm/`
  - `data/raw/polyvore_outfit/`
- Preprocessing outputs go to `data/processed/...`
- Configs in `configs/` control classes and paths.
"""

gitignore = """\
data/raw/
data/processed/
outputs/
models/
*.ckpt
*.onnx
*.pt
.ipynb_checkpoints/
__pycache__/
"""

def write_yaml(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)

def main():
    for d in DIRS:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    write_yaml(PROJECT_ROOT / "configs/labels.yaml", labels_yaml)
    write_yaml(PROJECT_ROOT / "configs/paths.yaml", paths_yaml)

    write_text(PROJECT_ROOT / "env/requirements.txt", requirements)
    write_yaml(PROJECT_ROOT / "env/conda_env.yaml", conda_env)

    write_text(PROJECT_ROOT / "README.md", readme)
    write_text(PROJECT_ROOT / ".gitignore", gitignore)

    print("✅ Project scaffold created at:", PROJECT_ROOT)
    print("• Update configs/paths.yaml if you move folders.")
    print("• Place datasets in data/raw/… and we’ll preprocess next.")

if __name__ == "__main__":
    main()
