from pathlib import Path
import sqlite3
import uuid
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from app.xgb_ranker import build_feature_frame, build_training_data, train_xgb, load_xgb, model_exists

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data"
DB_PATH = PROJECT_ROOT / "fashion_ai.db"
STYLES_PATH = DATA_DIR / "styles.csv"
EMBEDDINGS_PATH = DATA_DIR / "item_embeddings.npy"
ITEM_IDS_PATH = DATA_DIR / "item_ids.npy"

IMAGES_PATH = PROJECT_ROOT / "backend" / "images"
if not IMAGES_PATH.exists():
    IMAGES_PATH = PROJECT_ROOT / "images"

STYLE_UPLOAD_DIR = PROJECT_ROOT / "backend" / "style_uploads"
STYLE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Fashion Curator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if IMAGES_PATH.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_PATH)), name="images")
app.mount("/style-uploads", StaticFiles(directory=str(STYLE_UPLOAD_DIR)), name="style_uploads")

_text_model = None
_df = None
_item_embeddings = None
_options_cache = None


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS preferences (
        user_id TEXT PRIMARY KEY,
        gender TEXT,
        preferred_colors TEXT,
        disliked_colors TEXT,
        preferred_categories TEXT,
        preferred_types TEXT,
        preferred_usage TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        item_id INTEGER,
        action TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS visual_profiles (
        user_id TEXT PRIMARY KEY,
        learned_colors TEXT,
        brightness_label TEXT,
        contrast_label TEXT,
        vibe_label TEXT,
        image_url TEXT
    )
    """)

    conn.commit()
    conn.close()


def create_user_if_needed(user_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
    conn.commit()
    conn.close()


def load_preferences(user_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT gender, preferred_colors, disliked_colors, preferred_categories, preferred_types, preferred_usage
        FROM preferences
        WHERE user_id = ?
    """, (user_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return {
            "gender": "All",
            "preferred_colors": [],
            "disliked_colors": [],
            "preferred_categories": [],
            "preferred_types": [],
            "preferred_usage": []
        }

    return {
        "gender": row[0] or "All",
        "preferred_colors": row[1].split(",") if row[1] else [],
        "disliked_colors": row[2].split(",") if row[2] else [],
        "preferred_categories": row[3].split(",") if row[3] else [],
        "preferred_types": row[4].split(",") if row[4] else [],
        "preferred_usage": row[5].split(",") if row[5] else []
    }


def load_visual_profile(user_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT learned_colors, brightness_label, contrast_label, vibe_label, image_url
        FROM visual_profiles
        WHERE user_id = ?
    """, (user_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return {
            "learned_colors": [],
            "brightness_label": "",
            "contrast_label": "",
            "vibe_label": "",
            "image_url": ""
        }

    return {
        "learned_colors": row[0].split(",") if row[0] else [],
        "brightness_label": row[1] or "",
        "contrast_label": row[2] or "",
        "vibe_label": row[3] or "",
        "image_url": row[4] or ""
    }


def get_feedback(user_id: str):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT item_id FROM interactions WHERE user_id = ? AND action = 'like'", (user_id,))
    liked_ids = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT item_id FROM interactions WHERE user_id = ? AND action = 'dislike'", (user_id,))
    disliked_ids = [r[0] for r in cur.fetchall()]

    conn.close()
    return liked_ids, disliked_ids


def load_dataframe():
    global _df, _options_cache
    if _df is not None:
        return _df

    if not STYLES_PATH.exists():
        raise FileNotFoundError(f"styles.csv not found at {STYLES_PATH}")

    df = pd.read_csv(STYLES_PATH, engine="python", on_bad_lines="skip")

    needed = [
        "id", "gender", "masterCategory", "subCategory", "articleType",
        "baseColour", "season", "usage", "productDisplayName"
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = ""

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)

    for col in needed:
        if col != "id":
            df[col] = df[col].fillna("").astype(str)

    df["item_text"] = (
        df["gender"] + " " +
        df["masterCategory"] + " " +
        df["subCategory"] + " " +
        df["articleType"] + " " +
        df["baseColour"] + " " +
        df["season"] + " " +
        df["usage"] + " " +
        df["productDisplayName"]
    ).str.lower()

    _df = df
    _options_cache = {
        "genders": ["All"] + sorted([x for x in df["gender"].unique().tolist() if x]),
        "colors": sorted([x for x in df["baseColour"].unique().tolist() if x]),
        "categories": sorted([x for x in df["masterCategory"].unique().tolist() if x]),
        "types": sorted([x for x in df["articleType"].unique().tolist() if x]),
        "usage": sorted([x for x in df["usage"].unique().tolist() if x]),
    }
    return _df


def get_options_cache():
    load_dataframe()
    return _options_cache


def get_text_model():
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer
        _text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _text_model


SYNONYMS = {
    "hoodie": "sweatshirt",
    "tee": "tshirt",
    "t-shirt": "tshirt",
    "pants": "trousers",
    "sneakers": "shoes",
    "trainers": "shoes",
}


def normalize_query(text: str) -> str:
    words = text.lower().split()
    words = [SYNONYMS.get(w, w) for w in words]
    return " ".join(words)


def get_item_embeddings():
    global _item_embeddings
    df = load_dataframe()

    if _item_embeddings is not None:
        return _item_embeddings

    if EMBEDDINGS_PATH.exists() and ITEM_IDS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        item_ids = np.load(ITEM_IDS_PATH)

        order = {int(item_id): idx for idx, item_id in enumerate(item_ids.tolist())}
        df["_embed_idx"] = df["id"].map(order)
        df.dropna(subset=["_embed_idx"], inplace=True)
        df["_embed_idx"] = df["_embed_idx"].astype(int)
        df.sort_values("_embed_idx", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.drop(columns=["_embed_idx"], inplace=True)

        _item_embeddings = embeddings
        return _item_embeddings

    model = get_text_model()
    _item_embeddings = model.encode(df["item_text"].tolist(), convert_to_numpy=True, show_progress_bar=False)
    return _item_embeddings


COMMON_COLORS = {
    "Black": (25, 25, 25),
    "White": (235, 235, 235),
    "Grey": (128, 128, 128),
    "Blue": (60, 100, 190),
    "Red": (190, 60, 60),
    "Green": (70, 140, 70),
    "Yellow": (220, 200, 60),
    "Brown": (140, 90, 50),
    "Pink": (220, 140, 170),
    "Purple": (140, 90, 180),
    "Orange": (220, 120, 40),
    "Beige": (210, 190, 150),
}


def nearest_color_name(rgb):
    best_name = "Grey"
    best_dist = None
    for name, ref in COMMON_COLORS.items():
        dist = sum((rgb[i] - ref[i]) ** 2 for i in range(3))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def analyze_style_image(file_path: Path):
    image = Image.open(file_path).convert("RGB").resize((160, 160))
    arr = np.asarray(image).reshape(-1, 3)

    brightness = float(arr.mean())
    contrast = float(arr.std())

    small = image.resize((60, 60)).quantize(colors=5).convert("RGB")
    colors = small.getcolors(maxcolors=3600) or []
    dominant_names = []
    for _, rgb in sorted(colors, key=lambda x: x[0], reverse=True):
        name = nearest_color_name(rgb)
        if name not in dominant_names:
            dominant_names.append(name)
    dominant_names = dominant_names[:3]

    if brightness < 85:
        brightness_label = "dark palette"
    elif brightness < 150:
        brightness_label = "balanced palette"
    else:
        brightness_label = "light palette"

    if contrast < 45:
        contrast_label = "low contrast"
    elif contrast < 70:
        contrast_label = "moderate contrast"
    else:
        contrast_label = "high contrast"

    if any(c in dominant_names for c in ["Black", "Grey", "White"]):
        vibe_label = "clean / minimal leaning"
    elif any(c in dominant_names for c in ["Blue", "Green"]):
        vibe_label = "casual / relaxed leaning"
    else:
        vibe_label = "bold / expressive leaning"

    return {
        "learned_colors": dominant_names,
        "brightness_label": brightness_label,
        "contrast_label": contrast_label,
        "vibe_label": vibe_label,
    }


class SearchRequest(BaseModel):
    user_id: str
    query: str


class PreferenceRequest(BaseModel):
    user_id: str
    gender: str = "All"
    preferred_colors: list[str] = []
    disliked_colors: list[str] = []
    preferred_categories: list[str] = []
    preferred_types: list[str] = []
    preferred_usage: list[str] = []


class FeedbackRequest(BaseModel):
    user_id: str
    item_id: int


def score_results(results: pd.DataFrame, prefs: dict, visual: dict, liked_ids: list[int], disliked_ids: list[int]):
    results = results.copy()
    results["pref_score"] = 0.0
    results["feedback_score"] = 0.0

    preferred_colors = list(dict.fromkeys((prefs["preferred_colors"] or []) + (visual["learned_colors"] or [])))

    if prefs["gender"] != "All":
        results.loc[results["gender"] == prefs["gender"], "pref_score"] += 1.0

    if preferred_colors:
        results.loc[results["baseColour"].isin(preferred_colors), "pref_score"] += 1.5

    if prefs["disliked_colors"]:
        results.loc[results["baseColour"].isin(prefs["disliked_colors"]), "pref_score"] -= 2.0

    if prefs["preferred_categories"]:
        results.loc[results["masterCategory"].isin(prefs["preferred_categories"]), "pref_score"] += 1.2

    if prefs["preferred_types"]:
        results.loc[results["articleType"].isin(prefs["preferred_types"]), "pref_score"] += 1.5

    if prefs["preferred_usage"]:
        results.loc[results["usage"].isin(prefs["preferred_usage"]), "pref_score"] += 1.0

    if liked_ids:
        results.loc[results["id"].isin(liked_ids), "feedback_score"] -= 5.0

    if disliked_ids:
        results.loc[results["id"].isin(disliked_ids), "feedback_score"] -= 10.0

    results["final_score"] = (results["semantic_score"] * 3.0 + results["pref_score"] + results["feedback_score"])
    return results.sort_values("final_score", ascending=False)


def rerank_for_you_with_xgb(results: pd.DataFrame, prefs: dict, visual: dict, liked_ids: list[int], disliked_ids: list[int], top_n: int):
    xgb_model, feature_cols = load_xgb()
    if xgb_model is None:
        results = score_results(results, prefs, visual, liked_ids, disliked_ids)
        results["xgb_score"] = results["final_score"]
        return results.head(top_n)

    feature_df, feature_cols = build_feature_frame(results, prefs, visual, liked_ids, disliked_ids)
    results = results.copy()

    manual = score_results(results.copy(), prefs, visual, liked_ids, disliked_ids)
    results["pref_score"] = manual["pref_score"].values
    results["feedback_score"] = manual["feedback_score"].values
    results["final_score"] = manual["final_score"].values

    results["xgb_score"] = xgb_model.predict_proba(feature_df[feature_cols])[:, 1]
    return results.sort_values("xgb_score", ascending=False).head(top_n)


@app.on_event("startup")
def startup_event():
    init_db()
    load_dataframe()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    return get_options_cache()


@app.get("/preferences/{user_id}")
def get_preferences(user_id: str):
    create_user_if_needed(user_id)
    return load_preferences(user_id)


@app.get("/style-profile/{user_id}")
def style_profile(user_id: str):
    create_user_if_needed(user_id)
    return load_visual_profile(user_id)


@app.get("/xgb-status")
def xgb_status():
    return {"trained": model_exists()}


@app.post("/train-xgb")
def train_xgb_model():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users")
    all_users = [r[0] for r in cur.fetchall()]
    conn.close()

    df = load_dataframe()
    training_df = build_training_data(
        df,
        all_users,
        load_preferences,
        load_visual_profile,
        get_feedback,
    )

    if training_df.empty or len(training_df["label"].unique()) < 2:
        return {"message": "Not enough interaction data to train XGBoost yet."}

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

    train_xgb(training_df, feature_cols)
    return {"message": f"XGBoost trained on {len(training_df)} rows."}


@app.post("/style-image/{user_id}")
async def upload_style_image(user_id: str, file: UploadFile = File(...)):
    create_user_if_needed(user_id)

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    file_name = f"{user_id}_{uuid.uuid4().hex[:8]}{suffix}"
    target = STYLE_UPLOAD_DIR / file_name

    content = await file.read()
    target.write_bytes(content)

    learned = analyze_style_image(target)
    image_url = f"/style-uploads/{file_name}"

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO visual_profiles (
            user_id, learned_colors, brightness_label, contrast_label, vibe_label, image_url
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            learned_colors=excluded.learned_colors,
            brightness_label=excluded.brightness_label,
            contrast_label=excluded.contrast_label,
            vibe_label=excluded.vibe_label,
            image_url=excluded.image_url
    """, (
        user_id,
        ",".join(learned["learned_colors"]),
        learned["brightness_label"],
        learned["contrast_label"],
        learned["vibe_label"],
        image_url
    ))
    conn.commit()
    conn.close()

    return {
        "message": "Style image uploaded.",
        "image_url": image_url,
        **learned
    }


@app.post("/preferences")
def save_preferences(req: PreferenceRequest):
    create_user_if_needed(req.user_id)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO preferences (
            user_id, gender, preferred_colors, disliked_colors,
            preferred_categories, preferred_types, preferred_usage
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            gender=excluded.gender,
            preferred_colors=excluded.preferred_colors,
            disliked_colors=excluded.disliked_colors,
            preferred_categories=excluded.preferred_categories,
            preferred_types=excluded.preferred_types,
            preferred_usage=excluded.preferred_usage
    """, (
        req.user_id,
        req.gender,
        ",".join(req.preferred_colors),
        ",".join(req.disliked_colors),
        ",".join(req.preferred_categories),
        ",".join(req.preferred_types),
        ",".join(req.preferred_usage),
    ))
    conn.commit()
    conn.close()
    return {"message": "saved"}


@app.post("/search")
def search(req: SearchRequest):
    create_user_if_needed(req.user_id)

    df = load_dataframe()
    item_embeddings = get_item_embeddings()
    model = get_text_model()

    prefs = load_preferences(req.user_id)
    visual = load_visual_profile(req.user_id)
    liked_ids, disliked_ids = get_feedback(req.user_id)

    q = normalize_query(req.query.strip())
    if not q:
        return []

    query_emb = model.encode([q], convert_to_numpy=True, show_progress_bar=False)
    sims = cosine_similarity(query_emb, item_embeddings).flatten()

    results = df.copy()
    results["semantic_score"] = sims
    results = results[results["semantic_score"] > 0.15].copy()

    if prefs["gender"] != "All":
        results = results[results["gender"] == prefs["gender"]].copy()

    results = score_results(results, prefs, visual, liked_ids, disliked_ids).head(20)
    results["xgb_score"] = results["final_score"]

    return results[[
        "id", "productDisplayName", "gender", "masterCategory",
        "subCategory", "articleType", "baseColour", "usage",
        "semantic_score", "pref_score", "feedback_score", "final_score", "xgb_score"
    ]].replace({np.nan: None}).to_dict(orient="records")


@app.post("/like")
def like(req: FeedbackRequest):
    create_user_if_needed(req.user_id)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO interactions (user_id, item_id, action) VALUES (?, ?, 'like')", (req.user_id, req.item_id))
    conn.commit()
    conn.close()

    return {"message": "liked"}


@app.post("/dislike")
def dislike(req: FeedbackRequest):
    create_user_if_needed(req.user_id)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO interactions (user_id, item_id, action) VALUES (?, ?, 'dislike')", (req.user_id, req.item_id))
    conn.commit()
    conn.close()

    return {"message": "disliked"}


@app.get("/recommend/{user_id}")
def recommend(user_id: str):
    create_user_if_needed(user_id)

    df = load_dataframe()
    item_embeddings = get_item_embeddings()
    model = get_text_model()

    prefs = load_preferences(user_id)
    visual = load_visual_profile(user_id)
    liked_ids, disliked_ids = get_feedback(user_id)

    has_profile = any([
        prefs["preferred_colors"],
        prefs["preferred_categories"],
        prefs["preferred_types"],
        prefs["preferred_usage"],
        prefs["disliked_colors"],
        visual["learned_colors"],
    ])
    has_feedback = bool(liked_ids or disliked_ids)

    if not has_profile and not has_feedback:
        return []

    query_parts = []
    query_parts.extend(prefs["preferred_colors"])
    query_parts.extend(visual["learned_colors"])
    query_parts.extend(prefs["preferred_categories"])
    query_parts.extend(prefs["preferred_types"])
    query_parts.extend(prefs["preferred_usage"])
    query = " ".join(query_parts).strip()

    if not query and liked_ids:
        liked_rows = df[df["id"].isin(liked_ids)]
        query = " ".join(
            liked_rows["articleType"].astype(str).tolist() +
            liked_rows["baseColour"].astype(str).tolist() +
            liked_rows["masterCategory"].astype(str).tolist()
        ).strip()

    if not query:
        return []

    query_emb = model.encode([normalize_query(query)], convert_to_numpy=True, show_progress_bar=False)
    sims = cosine_similarity(query_emb, item_embeddings).flatten()

    results = df.copy()
    results["semantic_score"] = sims
    results = results[~results["id"].isin(liked_ids + disliked_ids)].copy()
    results = results[results["semantic_score"] > 0.10].copy()

    if prefs["gender"] != "All":
        results = results[results["gender"] == prefs["gender"]].copy()

    results = rerank_for_you_with_xgb(results, prefs, visual, liked_ids, disliked_ids, top_n=12)

    return results[[
        "id", "productDisplayName", "gender", "masterCategory",
        "subCategory", "articleType", "baseColour", "usage",
        "semantic_score", "pref_score", "feedback_score", "final_score", "xgb_score"
    ]].replace({np.nan: None}).to_dict(orient="records")
