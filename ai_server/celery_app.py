from celery import Celery
import os

# Docker 사용 시 'redis', 로컬 테스트 시 'localhost'
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ai_server_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=False,
    # ai_server 폴더 안의 tasks 모듈을 찾도록 설정
    imports=["ai_server.tasks"]
)