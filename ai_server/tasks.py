import os
import sqlite3
import boto3
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_server.celery_app import celery_app
from ai_server.inference import detector


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


DATABASE_URL = os.path.join(BASE_DIR, "backend", "test.db")


s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    region_name=os.getenv("S3_REGION")
)
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

@celery_app.task(name="ai_server.tasks.run_analysis")
def run_analysis(job_id: str, s3_key: str):
    print(f"[Worker] 작업 수신: {job_id}")
    local_video_path = os.path.join(BASE_DIR, f"temp_{job_id}.mp4")
    
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    try:
        cursor.execute("UPDATE jobs SET status = 'PROCESSING' WHERE id = ?", (job_id,))
        conn.commit()
        
        print(f"[Worker] 영상 다운로드 중...")
        s3.download_file(BUCKET_NAME, s3_key, local_video_path)
        
        result = detector.predict(local_video_path)
        

        cursor.execute("""
            UPDATE jobs 
            SET status = 'COMPLETED', 
                is_abnormal = ?, 
                risk_level = ?,
                risk_color = ?,
                max_score = ?,
                start_time_sec = ?, 
                result_text = ?, 
                result_video_url = ?
            WHERE id = ?
        """, (
            result['is_abnormal'], 
            result.get('risk_level', '알 수 없음'),
            result.get('risk_color', 'secondary'),
            result.get('max_score', 0.0),
            result['start_time_sec'], 
            result['result_text'], 
            s3_key, 
            job_id
        ))
        conn.commit()
        print(f"[Worker] 분석 완료: {job_id}")

    except Exception as e:
        print(f"[Worker] 에러 발생: {e}")
        cursor.execute("UPDATE jobs SET status = 'FAILED' WHERE id = ?", (job_id,))
        conn.commit()
    
    finally:
        conn.close()
        if os.path.exists(local_video_path):
            os.remove(local_video_path)