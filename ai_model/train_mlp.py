#MLP단 train
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
from tqdm import tqdm

# 2/3단계 파일 임포트
from data_preprocessing.utils_data_loader import get_graph_for_part, BodyPartDataset, JOINT_MAP

# 사전학습 모델 임포트 (경로 주의)
# from models.stg_nf import STG_NF_Model
from main_train import Dummy_STG_NF_Model as STG_NF_Model # 임시 모델 사용

# --- (임시) MLP 모델 정의 (실제 모델로 대체) ---
class MLPClassifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # (예시: 간단한 2-layer MLP)
        self.fc1 = nn.Linear(feature_dim, feature_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feature_dim // 2, 1) # 이진 분류 (출력 1개)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)) # Sigmoid로 0~1 확률 출력
# ---------------------------------------------

# --- 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30 # 예시
LEARNING_RATE = 1e-3
# MLP 학습용 데이터 파일
NORMAL_MLP_PKL = "normal_train_mlp_processed.pkl"
ABNORMAL_MLP_PKL = "abnormal_train_mlp_processed.pkl"
# 사전학습된 STG-NF 모델 경로
PRETRAINED_DIR = "pretrained_stgnf"
# MLP 모델 저장 경로
MLP_SAVE_DIR = "finetuned_mlp"
os.makedirs(MLP_SAVE_DIR, exist_ok=True)

# --- 특징 추출 함수 ---
def extract_features(stgnf_model, dataloader, device):
    stgnf_model.eval() # 평가 모드
    all_features = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting features"):
            data = data.to(device)
            # [!!!] 실제 STG-NF 모델의 특징 추출 로직으로 대체 필요 [!!!]
            # model(data)가 (B, FeatureDim) 형태의 텐서를 반환해야 함
            features = stgnf_model(data) # 임시 모델은 스칼라 반환하므로 수정 필요
            # (임시 특징: 마지막 노드의 평균값 사용)
            if isinstance(features, torch.Tensor) and features.dim() > 0: # 모델 출력이 텐서일 때
                 features = data.mean(dim=(1, 2))[:, 0] # (B, C, T, V) -> (B, V) -> (B,) 임시 특징
            else: # 임시 모델처럼 스칼라일 때
                 features = torch.randn(data.size(0)).to(device) * features # (B,) 임시 특징

            all_features.append(features.cpu())
    return torch.cat(all_features)

# --- MLP 학습 함수 (Binary Cross Entropy 사용) ---
def train_mlp_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    accuracy = 100 * correct / total
    return total_loss / len(loader), accuracy

# --- 메인 실행 ---
if __name__ == "__main__":
    # 필요한 pkl 파일 확인
    if not (os.path.exists(NORMAL_MLP_PKL) and os.path.exists(ABNORMAL_MLP_PKL)):
        print(f"오류: '{NORMAL_MLP_PKL}' 또는 '{ABNORMAL_MLP_PKL}' 파일 없음.")
        exit()

    # 모든 부위에 대해 학습
    for part_name in JOINT_MAP.keys():
        print(f"\n--- [{part_name}] MLP 파인튜닝 시작 ---")

        # 1. 부위별 그래프(A) 생성 (STG-NF 모델 로드에 필요)
        A = get_graph_for_part(part_name, strategy='spatial', max_hop=1)
        num_nodes = A.shape[1]

        # 2. 사전학습된 STG-NF 모델 로드 및 Freeze
        stgnf_model_path = os.path.join(PRETRAINED_DIR, f"stgnf_pretrained_{part_name}.pth")
        if not os.path.exists(stgnf_model_path):
            print(f"경고: 사전학습 모델 '{stgnf_model_path}' 없음. 건너<0xEB><0x9C><0x85>니다.")
            continue
            
        stgnf_model = STG_NF_Model(in_channels=3, num_nodes=num_nodes, A=A).to(DEVICE)
        stgnf_model.load_state_dict(torch.load(stgnf_model_path, map_location=DEVICE))
        stgnf_model.eval()
        for param in stgnf_model.parameters():
            param.requires_grad = False
        print(f"  -> 사전학습 모델 로드 완료: {stgnf_model_path}")

        # 3. 정상/이상 데이터셋 및 로더 준비 (특징 추출용)
        normal_dataset = BodyPartDataset(NORMAL_MLP_PKL, part_name)
        abnormal_dataset = BodyPartDataset(ABNORMAL_MLP_PKL, part_name)
        # 특징 추출 시에는 shuffle=False
        normal_loader = DataLoader(normal_dataset, batch_size=BATCH_SIZE, shuffle=False)
        abnormal_loader = DataLoader(abnormal_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 4. 특징 추출
        print("  -> 정상 데이터 특징 추출 중...")
        normal_features = extract_features(stgnf_model, normal_loader, DEVICE)
        print("  -> 이상 데이터 특징 추출 중...")
        abnormal_features = extract_features(stgnf_model, abnormal_loader, DEVICE)
        
        # 특징 벡터와 레이블 생성 (정상=0, 이상=1)
        all_features = torch.cat([normal_features, abnormal_features])
        all_labels = torch.cat([
            torch.zeros(len(normal_features)),
            torch.ones(len(abnormal_features))
        ]).unsqueeze(1) # (N, 1) 형태로
        
        feature_dim = all_features.shape[1] # STG-NF에서 추출된 특징 차원
        print(f"  -> 특징 추출 완료. 총 {len(all_features)}개 샘플, 특징 차원: {feature_dim}")

        # 5. MLP 학습용 데이터셋 및 로더
        mlp_dataset = TensorDataset(all_features, all_labels)
        mlp_loader = DataLoader(mlp_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 6. MLP 모델 및 옵티마이저, 손실 함수
        mlp_model = MLPClassifier(feature_dim).to(DEVICE)
        optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss() # 이진 분류 손실 (Binary Cross Entropy)

        # 7. MLP 학습 루프
        print("  -> MLP 모델 학습 시작...")
        for epoch in range(1, EPOCHS + 1):
            loss, acc = train_mlp_epoch(mlp_model, mlp_loader, optimizer, criterion, DEVICE)
            if epoch % 5 == 0:
                 print(f"    Epoch {epoch:02d}/{EPOCHS}, Train Loss: {loss:.4f}, Accuracy: {acc:.2f}%")

        # 8. MLP 모델 저장
        save_path = os.path.join(MLP_SAVE_DIR, f"mlp_finetuned_{part_name}.pth")
        torch.save(mlp_model.state_dict(), save_path)
        print(f"-> [{part_name}] MLP 모델 저장 완료: {save_path}")