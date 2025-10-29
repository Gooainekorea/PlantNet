import torch

print(f"CUDA 사용 가능? {torch.cuda.is_available()}") # True or False
print(f"GPU 있음?: {torch.cuda.device_count()}") # 0 or 1

if torch.cuda.is_available():
    print(f"GPU 장치 번호: {torch.cuda.get_device_name(0)}") # GPU가 없으면 None 반환