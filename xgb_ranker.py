from pathlib import Path
import sqlite3
import joblib
import pandas as pd
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "backend" / "data"
MODEL_PATH = MODEL_DIR / "xgb_model.joblib"


def build_feature_frame(results: pd.DataFrame, prefs: dict, visual: dict, liked_ids: list[int], disliked_ids: list[int]):
    df = results.copy()

    preferred_colors = set((prefs.get("preferred_colors") or []) + (visual.get("learned_colors") or []))
    disliked_colors = set(prefs.get("disliked_colors") or [])
    preferred_categories = set(prefs.get("preferred_categories") or [])
    preferred_types = set(prefs.get("preferred_types") or [])
    preferred_usage = set(prefs.get("preferred_usage") or [])

    if prefs.get("gender") and prefs.get("gender") != "All":
        df["gender_match"] = (df["gender"] == prefs.get("gender")).astype(int)
    else:
        df["gender_match"] = 0

    df["preferred_color_match"] = df["baseColour"].isin(preferred_colors).astype(int)
    df["disliked_color_match"] = df["baseColour"].isin(disliked_colors).astype(int)
    df["preferred_category_match"] = df["masterCategory"].isin(preferred_categories).astype(int)
    df["preferred_type_match"] = df["articleType"].isin(preferred_types).astype(int)
    df["preferred_usage_match"] = df["usage"].isin(preferred_usage).astype(int)
    df["visual_color_match"] = df["baseColour"].isin(set(visual.get("learned_colors") or [])).astype(int)
    df["was_liked_before"] = df["id"].isin(liked_ids).astype(int)
    df["was_disliked_before"] = df["id"].isin(disliked_ids).astype(int)

    feature_cols = [
        "semantic_score",
        "gender_match",
        "preferred_color_match",
        "disliked_color_match",
        "preferred_category_match",
        "preferred_type_match",
        "preferred_usage_match",
        "visual_color_match",
        "was_liked_before",
        "was_disliked_before",
    ]
    return df, feature_cols


def build_training_data(df_catalog: pd.DataFrame, all_user_ids: list[str], load_preferences_fn, load_visual_fn, get_feedback_fn):
    rows = []

    for user_id in all_user_ids:
        prefs = load_preferences_fn(user_id)
        visual = load_visual_fn(user_id)
        liked_ids, disliked_ids = get_feedback_fn(user_id)

        liked_set = set(liked_ids)
        disliked_set = set(disliked_ids)
        if not liked_set and not disliked_set:
            continue

        preferred_colors = set((prefs.get("preferred_colors") or []) + (visual.get("learned_colors") or []))
        avoided_colors = set(prefs.get("disliked_colors") or [])
        preferred_categories = set(prefs.get("preferred_categories") or [])
        preferred_types = set(prefs.get("preferred_types") or [])
        preferred_usage = set(prefs.get("preferred_usage") or [])

        for item_id, label in [(x, 1) for x in liked_ids] + [(x, 0) for x in disliked_ids]:
            row = df_catalog[df_catalog["id"] == item_id]
            if row.empty:
                continue
            item = row.iloc[0]

            # Simple training proxy: interactions are highly relevant to the user
            semantic_score = 1.0

            rows.append({
                "semantic_score": semantic_score,
                "gender_match": int(prefs.get("gender") not in [None, "", "All"] and item["gender"] == prefs.get("gender")),
                "preferred_color_match": int(item["baseColour"] in preferred_colors),
                "disliked_color_match": int(item["baseColour"] in avoided_colors),
                "preferred_category_match": int(item["masterCategory"] in preferred_categories),
                "preferred_type_match": int(item["articleType"] in preferred_types),
                "preferred_usage_match": int(item["usage"] in preferred_usage),
                "visual_color_match": int(item["baseColour"] in set(visual.get("learned_colors") or [])),
                "was_liked_before": int(item_id in liked_set),
                "was_disliked_before": int(item_id in disliked_set),
                "label": label,
            })

    return pd.DataFrame(rows)


def train_xgb(training_df: pd.DataFrame, feature_cols: list[str], label_col: str = "label"):
    model = XGBClassifier(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    model.fit(training_df[feature_cols], training_df[label_col])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    return model


def load_xgb():
    if not MODEL_PATH.exists():
        return None, None
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_cols"]


def model_exists():
    return MODEL_PATH.exists()
