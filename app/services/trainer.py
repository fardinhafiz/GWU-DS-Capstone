from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ..database import get_all_interactions, load_preferences
from .data_loader import get_text_embeddings, load_catalog

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / 'data' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / 'xgboost_ranker.joblib'
METRICS_PATH = MODEL_DIR / 'xgboost_metrics.joblib'

STYLE_TAG_COLUMNS = [
    'minimalist neutral streetwear',
    'sporty athleisure',
    'smart casual office wear',
    'formal occasion wear',
    'oversized relaxed fit',
    'bright colorful playful style',
    'monochrome clean essentials',
]


def _safe_mean_similarity(target_vector: np.ndarray, candidate_vectors: np.ndarray) -> float:
    if candidate_vectors.size == 0:
        return 0.0
    sims = candidate_vectors @ target_vector
    return float(np.mean(sims))


def build_training_frame() -> pd.DataFrame:
    interactions = pd.DataFrame(get_all_interactions())
    if interactions.empty:
        return pd.DataFrame()

    catalog = load_catalog().set_index('id')
    embeddings = get_text_embeddings()
    id_to_pos = {item_id: idx for idx, item_id in enumerate(catalog.index.tolist())}

    rows = []
    for user_id, user_df in interactions.groupby('user_id'):
        prefs = load_preferences(user_id)
        liked_ids = user_df.loc[user_df['action'] == 'like', 'item_id'].astype(int).tolist()
        disliked_ids = user_df.loc[user_df['action'] == 'dislike', 'item_id'].astype(int).tolist()

        liked_vectors = np.vstack([embeddings[id_to_pos[i]] for i in liked_ids if i in id_to_pos]) if liked_ids else np.empty((0, embeddings.shape[1]))
        disliked_vectors = np.vstack([embeddings[id_to_pos[i]] for i in disliked_ids if i in id_to_pos]) if disliked_ids else np.empty((0, embeddings.shape[1]))

        for _, row in user_df.iterrows():
            item_id = int(row['item_id'])
            if item_id not in id_to_pos or item_id not in catalog.index:
                continue
            item = catalog.loc[item_id]
            vector = embeddings[id_to_pos[item_id]]
            label = 1 if row['action'] == 'like' else 0
            feature_row = {
                'text_to_liked_sim': _safe_mean_similarity(vector, liked_vectors),
                'text_to_disliked_sim': _safe_mean_similarity(vector, disliked_vectors),
                'matches_preferred_color': float(item['baseColour'] in prefs['preferred_colors']),
                'matches_disliked_color': float(item['baseColour'] in prefs['disliked_colors']),
                'matches_preferred_category': float(item['masterCategory'] in prefs['preferred_categories']),
                'matches_preferred_type': float(item['articleType'] in prefs['preferred_types']),
                'matches_preferred_usage': float(item['usage'] in prefs['preferred_usage']),
                'label': label,
            }
            for tag in STYLE_TAG_COLUMNS:
                feature_row[f'style_tag::{tag}'] = float(tag in prefs.get('style_tags', []))
            rows.append(feature_row)

    return pd.DataFrame(rows)


def train_xgboost_model() -> dict:
    frame = build_training_frame()
    if frame.empty or frame['label'].nunique() < 2 or len(frame) < 12:
        return {
            'status': 'insufficient_data',
            'message': 'Need more like/dislike interactions before XGBoost can be trained.'
        }

    X = frame.drop(columns=['label'])
    y = frame['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        'model': 'xgboost',
        'f1': float(f1_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, probs)),
        'training_rows': int(len(frame)),
    }
    joblib.dump(model, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)
    return {'status': 'ok', 'metrics': metrics}


def load_xgboost_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return joblib.load(METRICS_PATH)
    return {}
