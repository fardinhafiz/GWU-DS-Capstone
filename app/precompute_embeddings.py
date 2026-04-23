from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data"
STYLES_PATH = DATA_DIR / "styles.csv"
EMBEDDINGS_PATH = DATA_DIR / "item_embeddings.npy"
ITEM_IDS_PATH = DATA_DIR / "item_ids.npy"

def main():
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

    print("Loading Sentence-BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Encoding {len(df):,} items...")
    embeddings = model.encode(
        df["item_text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(ITEM_IDS_PATH, df["id"].to_numpy())

    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
    print(f"Saved item ids to: {ITEM_IDS_PATH}")

if __name__ == "__main__":
    main()
