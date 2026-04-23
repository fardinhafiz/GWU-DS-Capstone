from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

FASHION_SYNONYMS = {
    'hoodie': 'sweatshirt',
    'hoodies': 'sweatshirts',
    'tee': 't-shirt',
    'tees': 't-shirts',
    'trainers': 'sneakers',
    'kicks': 'sneakers',
    'trousers': 'pants',
    'slacks': 'pants',
    'pullover': 'sweater',
    'pullovers': 'sweaters',
    'joggers': 'track pants',
    'formal shoes': 'dress shoes',
    'coat': 'jacket',
}


@lru_cache(maxsize=1)
def get_text_model() -> SentenceTransformer:
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@lru_cache(maxsize=1)
def get_style_label_embeddings() -> tuple[list[str], np.ndarray]:
    labels = [
        'minimalist neutral streetwear',
        'sporty athleisure',
        'smart casual office wear',
        'formal occasion wear',
        'oversized relaxed fit',
        'bright colorful playful style',
        'monochrome clean essentials',
    ]
    embeddings = embed_texts(labels)
    return labels, embeddings


def normalize_fashion_text(text: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9\s-]', ' ', text.lower())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    tokens = []
    for token in cleaned.split():
        tokens.append(FASHION_SYNONYMS.get(token, token))
    normalized = ' '.join(tokens)
    for phrase, replacement in FASHION_SYNONYMS.items():
        normalized = normalized.replace(phrase, replacement)
    return normalized


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_text_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([normalize_fashion_text(query)])[0]


def infer_style_tags_from_text(text: str, top_n: int = 2) -> list[str]:
    normalized = normalize_fashion_text(text)
    query_embedding = embed_texts([normalized])[0]
    labels, label_embeddings = get_style_label_embeddings()
    scores = label_embeddings @ query_embedding
    order = np.argsort(scores)[::-1][:top_n]
    return [labels[idx] for idx in order if scores[idx] > 0.2]
