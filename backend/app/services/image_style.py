from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@lru_cache(maxsize=1)
def get_clip_components() -> tuple[CLIPProcessor, CLIPModel]:
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    model.eval()
    return processor, model


def embed_image(image_path: str | Path) -> np.ndarray:
    processor, model = get_clip_components()
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt')
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy()


def average_embeddings(paths: list[str]) -> np.ndarray | None:
    vectors = []
    for path in paths:
        try:
            vectors.append(embed_image(path))
        except Exception:
            continue
    if not vectors:
        return None
    matrix = np.vstack(vectors)
    centroid = matrix.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return centroid
    return centroid / norm
