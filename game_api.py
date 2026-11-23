import base64
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
# Khi đóng gói exe, DB nằm cạnh file chạy để dữ liệu không mất sau mỗi lần mở
DATA_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else BASE_DIR
DB_PATH = str(DATA_DIR / "green_warrior.db")
MODEL_PATH = BASE_DIR / "waste_classifier_multiclass.h5"
CLASS_INDICES_PATH = BASE_DIR / "class_indices.json"
UI_PATH = BASE_DIR / "game_ui.html"

app = FastAPI(title="Green Warrior API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            score INTEGER DEFAULT 0,
            level INTEGER DEFAULT 1,
            exp INTEGER DEFAULT 0,
            collection TEXT DEFAULT '[]'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            label TEXT,
            confidence REAL,
            timestamp DATETIME,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS achievements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            achievement_title TEXT,
            unlocked_at DATETIME,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


init_db()


def ensure_user(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
    conn.close()


model = None
class_names = None
try:
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    class_names = [None] * len(class_indices)
    for name, idx in class_indices.items():
        class_names[idx] = name
    print("--- AI Engine & Database đã sẵn sàng! ---")
except Exception as e:
    print(f"Lỗi khởi tạo mô hình: {e}")


class ImagePayload(BaseModel):
    image: str
    username: str = "Ẩn danh"


class UserSync(BaseModel):
    username: str
    score: int
    level: int
    exp: int
    collection: list


@app.get("/")
def serve_game():
    if not UI_PATH.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy game_ui.html")
    return FileResponse(UI_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/login")
async def login(username: str):
    ensure_user(username)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return {
        "username": user[0],
        "score": user[1],
        "level": user[2],
        "exp": user[3],
        "collection": json.loads(user[4]),
    }


@app.post("/sync")
async def sync_data(data: UserSync):
    try:
        ensure_user(data.username)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE users
            SET score = ?, level = ?, exp = ?, collection = ?
            WHERE username = ?
        """,
            (data.score, data.level, data.exp, json.dumps(data.collection), data.username),
        )
        conn.commit()
        conn.close()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/predict")
async def predict(data: ImagePayload):
    if model is None or class_names is None:
        raise HTTPException(status_code=503, detail="Model chưa được tải")

    try:
        ensure_user(data.username)

        # Ảnh gửi lên ở dạng data URL: "data:image/jpeg;base64,...."
        encoded_data = data.image.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không thể giải mã ảnh")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Cắt giữa ảnh thành hình vuông để vật thể không bị méo
        h, w, _ = img_rgb.shape
        size = min(h, w)
        start_h, start_w = (h - size) // 2, (w - size) // 2
        img_cropped = img_rgb[start_h:start_h + size, start_w:start_w + size]

        img_resized = cv2.resize(img_cropped, (224, 224))
        img_array = img_resized.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)[0]
        idx = np.argmax(preds)
        label = class_names[idx]
        confidence = float(preds[idx])

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO scans (username, label, confidence, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (data.username, label, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        return {"label": label, "confidence": confidence, "status": "success"}
    except Exception as e:
        print(f"Predict error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/history/{username}")
async def get_history(username: str):
    ensure_user(username)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT label, confidence, timestamp FROM scans WHERE username = ? ORDER BY timestamp DESC LIMIT 20",
        (username,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"label": r[0], "confidence": r[1], "timestamp": r[2]}
        for r in rows
    ]


@app.get("/leaderboard")
async def get_leaderboard():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username, score, level FROM users ORDER BY score DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    return [
        {"username": r[0], "score": r[1], "level": r[2]}
        for r in rows
    ]


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
