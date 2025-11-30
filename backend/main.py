from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import uuid
import sqlite3
import boto3
from dotenv import load_dotenv
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=DOTENV_PATH) 


sys.path.append(BASE_DIR) 
from ai_server.tasks import run_analysis  

# 설정 
app = FastAPI()
templates = Jinja2Templates(directory="backend/templates")

DATABASE_URL = os.path.join(os.path.dirname(__file__), "test.db")

# S3 설정
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION")

s3 = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION
)


# API 엔드포인트

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_video(video_file: UploadFile = File(...)):
    #영상 업로드 및 AI 작업 지시 API (S3 연동)
    if not video_file.filename:
        print("S3 업로드 실패: 파일 이름이 없습니다.")
        raise HTTPException(status_code=400, detail="업로드 실패: 파일 이름이 없습니다.")

    job_id = str(uuid.uuid4())
    s3_video_path = f"uploads/{job_id}_{video_file.filename}"
    
    try:
        s3.upload_fileobj(
            video_file.file,
            S3_BUCKET_NAME,
            s3_video_path
        )
        print(f"S3 업로드 성공: {s3_video_path}")
    except Exception as e:
        print(f"S3 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"S3 업로드에 실패했습니다: {e}")

    # DB에 작업 기록 (PENDING 상태)
    conn = sqlite3.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO jobs (id, video_path, status, is_abnormal) VALUES (?, ?, ?, ?)",
        (job_id, s3_video_path, 'PENDING', False)
    )
    conn.commit()
    conn.close()

    run_analysis.delay(job_id, s3_video_path)  
    
    print(f"작업 접수 완료: {job_id}")
    return {"message": "업로드 성공. 분석이 시작되었습니다.", "job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    job = cursor.fetchone()
    conn.close()
    
    if not job:
        raise HTTPException(status_code=404, detail="작업 ID를 찾을 수 없습니다.")
    
    if job['status'] != 'COMPLETED':
        return {
            "status": job['status'], 
            "result_url": None, 
            "is_abnormal": False, 
            "start_time": 0.0
        }

    # S3 임시 URL 발급 로직
    video_url = None
    if job['result_video_url']:
        try:
            video_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET_NAME, 'Key': job['result_video_url']},
                ExpiresIn=3600  # 1시간
            )
        except Exception as e:
            print(f"S3 URL 생성 실패: {e}")
            video_url = None

    return {
        "status": job['status'], 
        "result_url": video_url, 
        "is_abnormal": job['is_abnormal'],
        "risk_level": job['risk_level'] if 'risk_level' in job.keys() else '알 수 없음',
        "risk_color": job['risk_color'] if 'risk_color' in job.keys() else 'secondary',
        "max_score": job['max_score'] if 'max_score' in job.keys() else 0.0,
        "result_text": job['result_text'],
        "start_time": job['start_time_sec']
    }