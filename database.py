import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / 'fashion_ai.db'


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )
            '''
        )
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS preferences (
                user_id TEXT PRIMARY KEY,
                gender TEXT,
                preferred_colors TEXT,
                disliked_colors TEXT,
                preferred_categories TEXT,
                preferred_types TEXT,
                preferred_usage TEXT,
                style_tags TEXT,
                updated_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
            '''
        )
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                source TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
            '''
        )
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS uploaded_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                embedding_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
            '''
        )


def ensure_user(user_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            'INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)',
            (user_id, datetime.utcnow().isoformat()),
        )


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in value.split(',') if part]


DEFAULT_PROFILE = {
    'gender': 'All',
    'preferred_colors': [],
    'disliked_colors': [],
    'preferred_categories': [],
    'preferred_types': [],
    'preferred_usage': [],
    'style_tags': [],
}


def load_preferences(user_id: str) -> dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            '''
            SELECT gender, preferred_colors, disliked_colors, preferred_categories,
                   preferred_types, preferred_usage, style_tags
            FROM preferences
            WHERE user_id = ?
            ''',
            (user_id,),
        ).fetchone()
    if row is None:
        return DEFAULT_PROFILE.copy()
    return {
        'gender': row['gender'] or 'All',
        'preferred_colors': _split_csv(row['preferred_colors']),
        'disliked_colors': _split_csv(row['disliked_colors']),
        'preferred_categories': _split_csv(row['preferred_categories']),
        'preferred_types': _split_csv(row['preferred_types']),
        'preferred_usage': _split_csv(row['preferred_usage']),
        'style_tags': _split_csv(row['style_tags']),
    }


def save_preferences(user_id: str, payload: dict[str, Any]) -> None:
    ensure_user(user_id)
    now = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            '''
            INSERT INTO preferences (
                user_id, gender, preferred_colors, disliked_colors,
                preferred_categories, preferred_types, preferred_usage,
                style_tags, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                gender = excluded.gender,
                preferred_colors = excluded.preferred_colors,
                disliked_colors = excluded.disliked_colors,
                preferred_categories = excluded.preferred_categories,
                preferred_types = excluded.preferred_types,
                preferred_usage = excluded.preferred_usage,
                style_tags = excluded.style_tags,
                updated_at = excluded.updated_at
            ''',
            (
                user_id,
                payload.get('gender', 'All'),
                ','.join(payload.get('preferred_colors', [])),
                ','.join(payload.get('disliked_colors', [])),
                ','.join(payload.get('preferred_categories', [])),
                ','.join(payload.get('preferred_types', [])),
                ','.join(payload.get('preferred_usage', [])),
                ','.join(payload.get('style_tags', [])),
                now,
            ),
        )


def save_interaction(user_id: str, item_id: int, action: str, source: str | None = None) -> None:
    ensure_user(user_id)
    with get_conn() as conn:
        conn.execute(
            '''
            INSERT INTO interactions (user_id, item_id, action, source, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (user_id, item_id, action, source or 'ui', datetime.utcnow().isoformat()),
        )


def get_user_item_ids(user_id: str, action: str) -> list[int]:
    with get_conn() as conn:
        rows = conn.execute(
            'SELECT item_id FROM interactions WHERE user_id = ? AND action = ?',
            (user_id, action),
        ).fetchall()
    return [int(row['item_id']) for row in rows]


def save_uploaded_image(user_id: str, file_path: str, embedding_path: str | None = None) -> None:
    ensure_user(user_id)
    with get_conn() as conn:
        conn.execute(
            '''
            INSERT INTO uploaded_images (user_id, file_path, embedding_path, created_at)
            VALUES (?, ?, ?, ?)
            ''',
            (user_id, file_path, embedding_path, datetime.utcnow().isoformat()),
        )


def get_uploaded_images(user_id: str) -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            'SELECT file_path, embedding_path, created_at FROM uploaded_images WHERE user_id = ? ORDER BY id DESC',
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_interactions() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            'SELECT user_id, item_id, action, source, timestamp FROM interactions ORDER BY id'
        ).fetchall()
    return [dict(row) for row in rows]
