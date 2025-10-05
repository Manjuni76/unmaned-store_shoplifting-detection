import torch
import warnings

print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")

# GPU 호환성 체크 함수
def check_gpu_compatibility():
    """GPU가 실제로 사용 가능한지 안전하게 확인"""
    if not torch.cuda.is_available():
        return False, "CUDA가 사용 불가능합니다."
    
    try:
        # 간단한 GPU 테스트
        device = torch.device('cuda:0')
        test_tensor = torch.tensor([1.0, 2.0]).to(device)
        result = test_tensor * 2
        result_cpu = result.cpu()
        return True, f"GPU 테스트 성공: {torch.cuda.get_device_name(0)}"
    except Exception as e:
        return False, f"GPU 호환성 문제: {str(e)}"

# GPU 호환성 확인
gpu_available, gpu_message = check_gpu_compatibility()
print(f"\nGPU 상태: {gpu_message}")

if gpu_available:
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CPU 모드로 실행됩니다.")

# 추천 디바이스 설정
def get_device():
    """안전한 디바이스 선택"""
    if gpu_available:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

recommended_device = get_device()
print(f"\n권장 디바이스: {recommended_device}")

# 디바이스 호환성 테스트
print("\n=== 디바이스 호환성 테스트 ===")
try:
    test_tensor = torch.randn(100, 100).to(recommended_device)
    result = torch.matmul(test_tensor, test_tensor.T)
    print(f"✅ {recommended_device} 에서 연산 테스트 성공")
    print(f"   결과 크기: {result.shape}")
    print(f"   결과 합계: {result.sum().item():.2f}")
except Exception as e:
    print(f"❌ {recommended_device} 에서 연산 테스트 실패: {e}")
    print("   CPU 모드를 사용하세요.")