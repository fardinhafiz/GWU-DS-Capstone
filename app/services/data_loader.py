from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .nlp import embed_texts, normalize_fashion_text

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
STYLES_PATH = DATA_DIR / 'styles.csv'
EMBEDDINGS_PATH = DATA_DIR / 'text_embeddings.npy'
IMAGE_DIR = BASE_DIR / 'images'
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.webp']


REQUIRED_COLUMNS = [
    'id', 'gender', 'masterCategory', 'subCategory', 'articleType',
    'baseColour', 'season', 'usage', 'productDisplayName'
]


def find_image_path(pid: int) -> str | None:
    for ext in IMAGE_EXTS:
        candidate = IMAGE_DIR / f'{pid}{ext}'
        if candidate.exists():
            return str(candidate)
    return None


@lru_cache(maxsize=1)
def load_catalog() -> pd.DataFrame:
    if not STYLES_PATH.exists():
        raise FileNotFoundError(
            f'Missing styles.csv at {STYLES_PATH}. Place your dataset in backend/data/styles.csv'
        )
    df = pd.read_csv(STYLES_PATH, engine='python', on_bad_lines='skip')
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ''
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df = df.dropna(subset=['id']).copy()
    df['id'] = df['id'].astype(int)
    for column in REQUIRED_COLUMNS:
        if column != 'id':
            df[column] = df[column].fillna('').astype(str)
    df['item_text'] = (
        df['gender'] + ' ' +
        df['masterCategory'] + ' ' +
        df['subCategory'] + ' ' +
        df['articleType'] + ' ' +
        df['baseColour'] + ' ' +
        df['season'] + ' ' +
        df['usage'] + ' ' +
        df['productDisplayName']
    ).map(normalize_fashion_text)
    df['image_path'] = df['id'].map(find_image_path)
    return df


@lru_cache(maxsize=1)
def get_text_embeddings() -> np.ndarray:
    df = load_catalog()
    if EMBEDDINGS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        if len(embeddings) == len(df):
            return embeddings
    embeddings = embed_texts(df['item_text'].tolist())
    np.save(EMBEDDINGS_PATH, embeddings)
    return embeddings
