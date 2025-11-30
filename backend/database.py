import sqlite3
import os

DATABASE_URL = os.path.join(os.path.dirname(__file__), "test.db")

def init_db():
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        video_path TEXT NOT NULL,
        status TEXT NOT NULL,
        result_text TEXT, 
        is_abnormal BOOLEAN DEFAULT FALSE,
        result_video_url TEXT,
        start_time_sec REAL DEFAULT 0.0  -- (!!!) "시작 시간(초)" 컬럼 추가 (!!!)
    );
    """)
    conn.commit()
    conn.close()
    print(f"Database v3 (with start_time) initialized at {DATABASE_URL}")

if __name__ == "__main__":
    init_db()