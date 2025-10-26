# 부위별 STG-NF 모델 학습 후 freeze
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# 2/3단계 파일 임포트
from data_preprocessing.utils_data_loader import get_graph_for_part, BodyPartDataset, JOINT_MAP

# (가정) 곽민준 님의 실제 STG-NF 모델 클래스 임포트
# from models.stg_nf import STG_NF_Model
from main_train import Dummy_STG_NF_Model as STG_NF_Model # 임시 모델 사용

# --- 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 50 # 예시
LEARNING_RATE = 1e-4
NORMAL_TRAIN_PKL = "normal_train_processed.pkl" # 1단계에서 생성된 파일
MODEL_SAVE_DIR = "pretrained_stgnf"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 학습 함수 (main_train.py에서 가져옴, Loss 수정 필요) ---
def train_stgnf_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader: # data: (B, C, T, V)
        data = data.to(device)
        optimizer.zero_grad()

        # [!!!] 실제 STG-NF 모델의 loss 계산 로직으로 대체 필요 [!!!]
        # NF 모델은 보통 log likelihood maximization (음수 로그 확률 최소화)
        # log_prob = model(data)
        # loss = -log_prob.mean()
        
        # (임시 Loss)
        pred_val = model(data)
        loss = torch.mean((pred_val - 0.5)**2) # 임시 목표값 0.5

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# --- 메인 실행 ---
if __name__ == "__main__":
    if not os.path.exists(NORMAL_TRAIN_PKL):
        print(f"오류: '{NORMAL_TRAIN_PKL}' 파일 없음. main_preprocessing.py를 먼저 실행하세요.")
        exit()

    # 모든 부위에 대해 학습
    for part_name in JOINT_MAP.keys():
        print(f"\n--- [{part_name}] STG-NF 사전학습 시작 ---")

        # 1. 부위별 그래프(A) 생성
        A = get_graph_for_part(part_name, strategy='spatial', max_hop=1)
        num_nodes = A.shape[1]

        # 2. 부위별 데이터셋 및 로더 (정상 데이터만 사용)
        dataset = BodyPartDataset(NORMAL_TRAIN_PKL, part_name)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        # 3. 모델 및 옵티마이저 생성
        model = STG_NF_Model(in_channels=3, num_nodes=num_nodes, A=A).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 4. 학습 루프
        for epoch in range(1, EPOCHS + 1):
            loss = train_stgnf_epoch(model, loader, optimizer, DEVICE)
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:03d}/{EPOCHS}, Train Loss: {loss:.6f}")

        # 5. 모델 저장
        save_path = os.path.join(MODEL_SAVE_DIR, f"stgnf_pretrained_{part_name}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"-> [{part_name}] 사전학습 모델 저장 완료: {save_path}")