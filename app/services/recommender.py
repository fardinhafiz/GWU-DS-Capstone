from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..database import get_uploaded_images, get_user_item_ids, load_preferences
from .data_loader import get_text_embeddings, load_catalog
from .image_style import average_embeddings, embed_image
from .nlp import embed_query, infer_style_tags_from_text
from .trainer import STYLE_TAG_COLUMNS, load_xgboost_model


def _get_profile(user_id: str) -> dict:
    return load_preferences(user_id)


def _get_visual_centroid(user_id: str) -> np.ndarray | None:
    uploads = get_uploaded_images(user_id)
    paths = [entry['file_path'] for entry in uploads if entry.get('file_path')]
    if not paths:
        return None
    return average_embeddings(paths)


def _style_overlap_score(item_text: str, style_tags: list[str]) -> float:
    if not style_tags:
        return 0.0
    lowered = item_text.lower()
    matches = sum(1 for tag in style_tags if any(part in lowered for part in tag.split()))
    return matches / max(len(style_tags), 1)


def _build_ml_features(user_id: str, candidate_ids: list[int]) -> pd.DataFrame:
    catalog = load_catalog().set_index('id')
    embeddings = get_text_embeddings()
    id_to_idx = {item_id: idx for idx, item_id in enumerate(catalog.index.tolist())}
    prefs = _get_profile(user_id)
    liked_ids = [item_id for item_id in get_user_item_ids(user_id, 'like') if item_id in id_to_idx]
    disliked_ids = [item_id for item_id in get_user_item_ids(user_id, 'dislike') if item_id in id_to_idx]

    liked_vectors = np.vstack([embeddings[id_to_idx[i]] for i in liked_ids]) if liked_ids else np.empty((0, embeddings.shape[1]))
    disliked_vectors = np.vstack([embeddings[id_to_idx[i]] for i in disliked_ids]) if disliked_ids else np.empty((0, embeddings.shape[1]))

    rows = []
    for item_id in candidate_ids:
        item = catalog.loc[item_id]
        vector = embeddings[id_to_idx[item_id]]
        row = {
            'text_to_liked_sim': float(np.mean(liked_vectors @ vector)) if liked_vectors.size else 0.0,
            'text_to_disliked_sim': float(np.mean(disliked_vectors @ vector)) if disliked_vectors.size else 0.0,
            'matches_preferred_color': float(item['baseColour'] in prefs['preferred_colors']),
            'matches_disliked_color': float(item['baseColour'] in prefs['disliked_colors']),
            'matches_preferred_category': float(item['masterCategory'] in prefs['preferred_categories']),
            'matches_preferred_type': float(item['articleType'] in prefs['preferred_types']),
            'matches_preferred_usage': float(item['usage'] in prefs['preferred_usage']),
        }
        for tag in STYLE_TAG_COLUMNS:
            row[f'style_tag::{tag}'] = float(tag in prefs.get('style_tags', []))
        rows.append(row)
    return pd.DataFrame(rows)


def search_catalog(user_id: str, query: str, top_k: int = 12) -> list[dict]:
    catalog = load_catalog().copy()
    embeddings = get_text_embeddings()
    query_vector = embed_query(query)

    semantic_scores = embeddings @ query_vector
    catalog['semantic_score'] = semantic_scores

    prefs = _get_profile(user_id)
    inferred_style_tags = infer_style_tags_from_text(query)
    combined_style_tags = list(dict.fromkeys(prefs.get('style_tags', []) + inferred_style_tags))

    catalog['preference_score'] = 0.0
    if prefs['gender'] != 'All':
        catalog.loc[catalog['gender'] == prefs['gender'], 'preference_score'] += 0.45
    if prefs['preferred_colors']:
        catalog.loc[catalog['baseColour'].isin(prefs['preferred_colors']), 'preference_score'] += 0.75
    if prefs['disliked_colors']:
        catalog.loc[catalog['baseColour'].isin(prefs['disliked_colors']), 'preference_score'] -= 1.2
    if prefs['preferred_categories']:
        catalog.loc[catalog['masterCategory'].isin(prefs['preferred_categories']), 'preference_score'] += 0.7
    if prefs['preferred_types']:
        catalog.loc[catalog['articleType'].isin(prefs['preferred_types']), 'preference_score'] += 0.8
    if prefs['preferred_usage']:
        catalog.loc[catalog['usage'].isin(prefs['preferred_usage']), 'preference_score'] += 0.6

    liked_ids = get_user_item_ids(user_id, 'like')
    disliked_ids = get_user_item_ids(user_id, 'dislike')
    id_to_idx = {item_id: idx for idx, item_id in enumerate(catalog['id'].tolist())}

    liked_vectors = np.vstack([embeddings[id_to_idx[i]] for i in liked_ids if i in id_to_idx]) if liked_ids else np.empty((0, embeddings.shape[1]))
    disliked_vectors = np.vstack([embeddings[id_to_idx[i]] for i in disliked_ids if i in id_to_idx]) if disliked_ids else np.empty((0, embeddings.shape[1]))

    if liked_vectors.size:
        catalog['liked_similarity'] = np.mean(embeddings @ liked_vectors.T, axis=1)
    else:
        catalog['liked_similarity'] = 0.0

    if disliked_vectors.size:
        catalog['disliked_similarity'] = np.mean(embeddings @ disliked_vectors.T, axis=1)
    else:
        catalog['disliked_similarity'] = 0.0

    catalog['style_score'] = catalog['item_text'].map(lambda text: _style_overlap_score(text, combined_style_tags))

    visual_centroid = _get_visual_centroid(user_id)
    if visual_centroid is not None:
        visual_scores = []
        for image_path in catalog['image_path'].tolist():
            if image_path:
                try:
                    image_vector = embed_image(image_path)
                    visual_scores.append(float(np.dot(image_vector, visual_centroid)))
                except Exception:
                    visual_scores.append(0.0)
            else:
                visual_scores.append(0.0)
        catalog['visual_score'] = visual_scores
    else:
        catalog['visual_score'] = 0.0

    catalog['base_score'] = (
        catalog['semantic_score'] * 3.0 +
        catalog['preference_score'] * 1.35 +
        catalog['liked_similarity'] * 1.5 -
        catalog['disliked_similarity'] * 1.2 +
        catalog['style_score'] * 1.0 +
        catalog['visual_score'] * 1.2
    )

    model = load_xgboost_model()
    if model is not None:
        feature_frame = _build_ml_features(user_id, catalog['id'].tolist())
        catalog['xgb_score'] = model.predict_proba(feature_frame)[:, 1]
        catalog['final_score'] = catalog['base_score'] + catalog['xgb_score'] * 2.0
    else:
        catalog['xgb_score'] = 0.0
        catalog['final_score'] = catalog['base_score']

    seen_ids = set(liked_ids + disliked_ids)
    results = catalog.sort_values('final_score', ascending=False)
    results = results[results['semantic_score'] > 0.05]
    if results.empty:
        results = catalog.sort_values('final_score', ascending=False)
    return _frame_to_payload(results.head(top_k), seen_ids)


def recommend_for_user(user_id: str, top_k: int = 12) -> list[dict]:
    prefs = _get_profile(user_id)
    liked_ids = get_user_item_ids(user_id, 'like')
    disliked_ids = get_user_item_ids(user_id, 'dislike')

    has_preferences = any([
        prefs.get('preferred_colors'),
        prefs.get('preferred_categories'),
        prefs.get('preferred_types'),
        prefs.get('preferred_usage'),
        prefs.get('style_tags'),
        prefs.get('disliked_colors'),
    ])
    has_feedback = bool(liked_ids or disliked_ids)

    if not has_preferences and not has_feedback:
        return []

    query_parts = []
    query_parts.extend(prefs.get('preferred_colors', []))
    query_parts.extend(prefs.get('preferred_categories', []))
    query_parts.extend(prefs.get('preferred_types', []))
    query_parts.extend(prefs.get('preferred_usage', []))
    query_parts.extend(prefs.get('style_tags', []))

    if liked_ids:
        catalog = load_catalog().set_index('id')
        liked_rows = catalog.loc[[item_id for item_id in liked_ids if item_id in catalog.index]]
        if not liked_rows.empty:
            query_parts.extend(liked_rows['articleType'].tolist())
            query_parts.extend(liked_rows['baseColour'].tolist())
            query_parts.extend(liked_rows['usage'].tolist())

    query = ' '.join([part for part in query_parts if part]).strip()
    if not query:
        return []

    ranked = search_catalog(user_id, query=query, top_k=max(top_k * 4, 24))
    seen_ids = set(liked_ids + disliked_ids)
    filtered = [item for item in ranked if item['id'] not in seen_ids]
    return filtered[:top_k]


def _frame_to_payload(frame: pd.DataFrame, seen_ids: set[int]) -> list[dict]:
    payload = []
    for _, row in frame.iterrows():
        payload.append(
            {
                'id': int(row['id']),
                'title': row['productDisplayName'],
                'productDisplayName': row['productDisplayName'],
                'gender': row['gender'],
                'category': row['masterCategory'],
                'subcategory': row['subCategory'],
                'articleType': row['articleType'],
                'color': row['baseColour'],
                'usage': row['usage'],
                'imagePath': row['image_path'],
                'imageUrl': f"/catalog-images/{Path(row['image_path']).name}" if row['image_path'] else None,
                'semanticScore': float(row['semantic_score']),
                'preferenceScore': float(row['preference_score']),
                'likedSimilarity': float(row['liked_similarity']),
                'dislikedSimilarity': float(row['disliked_similarity']),
                'styleScore': float(row['style_score']),
                'visualScore': float(row['visual_score']),
                'xgbScore': float(row['xgb_score']),
                'finalScore': float(row['final_score']),
                'alreadyRated': int(row['id']) in seen_ids,
            }
        )
    return payload
