"""
파일명: AlexNet_Weights.py

AlexNet_Weights 모델의 모든 가중치 동결해 이미지 처리 부분 학습 특성을 유지,
최종 분류 레이어의 클래스 수에 맞게 재설계해 전이 학습 구현.

단순히 모델 학습 방향이 아닌 성능 향상 측정을 위한 비교가 필요해 평가지수 추가

- Kornia 라이브러리를 활용한 GPU가속 데이터 증강을 구현해 효율을 높임
    학습을 위한 이미지 데이터 증강처리 - 뒤집기,회전,블러등 GPU에서 처리
    -> CPU 병목 현상을 크게 줄임

- 훈련 프로세스
    손실함수(class_weights) : 클래스 샘플 수에 반비례
    옵티마이저(Adam) : 학습률 감소, 가중치 감쇠
    DataLoader의 num_workers, pin_memory로 CPU에서 GPU로 데이터 전송 최적화
    평가지표를 이용해 모델 성능 모니터링, 정체나 과적합시 동결 해제 

torchmetrics 라이브러리 설치 (pip install torchmetrics, python -m pip install torchmetrics)
Macro Precision : 각 클래스 별로 예측값중 정답인 비율을 계산후 평균을 냄 average='macro'
Macro F1Score : Precision과 Recall의 조화평균
Balanced Accuracy : 클래스 정확도를 평균냄
Top-5 Accuracy : 다중 클래스에서 상위 5개 예측이 맞으면 정답으로 간주
Confusion Matrix (오차 행렬) : 학습 종료시 이미지로 저장

악 속도, 크기, 연산량을 잊었어

"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
# from PIL import Image # Pillow-SIMD 지원안함
import kornia.augmentation as K # GPU가속!!!!! 제발!!!! 아니 너무느려!!!!!!
import cv2
import os
import logging
import torch # pythorch pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
import torch.nn as nn # 신경망 구성요소 및 모듈 제공
from torchvision.models import alexnet, AlexNet_Weights # 사전학습 모델 불러옴
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from tqdm import tqdm #실시간 진행 막대그래프
from collections import Counter
from torchmetrics import MetricCollection # SaveModel처럼 클래스 만들필요없음
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy # 다중분류 평가 지표
import time 
from thop import profile, clever_format # 모델 플롭스 계산

# config
base_input_path = '' # 기본입력경로
input_path = f'{base_input_path}plantnet_300K/' # 데이터 폴더
output_path = f'{base_input_path}output_data/' # 출력결과 
images_path = f'{input_path}images_resized/' #이미지 경로

metadata = pd.read_json(f'{output_path}metadata/metadata.json')

with open(f'{output_path}metadata/species_names.json', 'r') as f:
    species_idx = json.load(f)

def show_image(path): # 이미지 보이기
    img = cv2.imread(f'{images_path}{path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def show_tensor_image(tensor):
    tensor = tensor.detach().numpy().transpose((1, 2, 0))
    plt.imshow(tensor)
    plt.axis('off')  # To hide axis values
    plt.show()

#--------------------------------------------------------------
plt.style.use('ggplot')

model_path = os.path.join(output_path, 'models')
os.makedirs(model_path, exist_ok=True)
best_model_path = os.path.join(model_path, 'best_model.pth')
log_path = f'{output_path}models/AlexNet_training_log.txt'
os.makedirs(f'{output_path}models/', exist_ok=True)

# 디바이스 설정 (GPU 사용 가능 시 GPU, 아니면 CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 로거 설정 (터미널 출력 기록)
logger = logging.getLogger('model_training')
logger.setLevel(logging.INFO)

# 기존 핸들러가 있으면 제거 (중복 출력 방지)
if logger.hasHandlers():
    logger.handlers.clear()

# 파일 핸들러 (파일에 저장)
file_handler = logging.FileHandler(log_path, encoding='utf-8')
file_formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 스트림 핸들러 (터미널 출력)
stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter('%(message)s') 
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)


# 이미지 전처리 및 증강처리----------------------------------------

simple_transform = transforms.Compose([ # CPU가 담당하는 변환부분
    transforms.Resize((224, 224)), # AlexNet 입력 크기에 맞추기
    transforms.ToTensor()          # PIL 이미지를 PyTorch 텐서로 변환
])

# 데이터셋 정의
# transform에 클래스의 인스턴스를 전달. CPU 변환만 적용
train_dataset = datasets.ImageFolder( # 트레이닝 데이터
    root=f'{images_path}train/',
    transform=simple_transform 
)

test_dataset = datasets.ImageFolder( # 테스트 데이터
    root=f'{images_path}test/',
    transform=simple_transform 
)

# ------------------------------------GPU가 담당할 증강 및 정규화 파이프라인 정의--
"""
GPU에서 실행할 이미지 증강 및 정규화 파이프라인
PyTorch의 nn.Sequential - 신경망 레이어를 컨테이너 모듈로 관리하는 클래스
증강처리 변환을 순차적으로 연결해 최종적으로 증강 및 정규화된 이미지 출력
"""
gpu_augmentation = nn.Sequential(
    # 사전 훈련 모델 사용시 반드시 해당 모델의 훈련에 사용됬던 것과 동일한 평균과 표준편차로 입력 이미지를 정규화 해야 한다.
    # 아니면 믿지못할 학습그래프가 나올것이다
    K.RandomHorizontalFlip(p=0.5), # 좌우 반전
    K.RandomVerticalFlip(p=0.5), # 상하반전
    K.RandomRotation(degrees=45.0, p=1.0), # 1~45 랜덤 회전
    K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.5), # 가우시안 블러
    # ImageNet 정규화 
    # 중요함 - 훈련된 사전학습 모델에 맞춰 입력 이미지 평균과 표준편차로 정규화
    K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
).to(device)

gpu_normalization = nn.Sequential(
    K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
).to(device) 


# 모델 정의 및 설정 --------------------------------------------------------------


# 클래스별 이미지 수 계산
num_classes = len(species_idx["data"]) # 고유한 클래스의 수를 결정
class_counts = Counter(train_dataset.targets) # 각 클래스에 속하는 샘플의 수를 계산

class FineTuneAlexNet(nn.Module):
    """
    AlexNet 모델을 기반으로 한 전이 학습용 신경망 클래스
    - 사전 훈련된 AlexNet 모델을 불러와 마지막 분류 레이어를 재설계
    - 고유 클래스 수 결정 및 계산
    - 가중 치 계산 및 손실 함수 정의
    - 모든 레이어 동결 후 마지막 레이어만 학습
    - 3에폭 지표 정체시 모델을 별도 파일로 저장 및 3개의 레이어 순차적으로 동결 해제 및 가중치 조정
    """
    def __init__(self, num_classes,class_counts, device):        
        """
        생성자(초기화 함수)
        초기 설정 및 모델 구성
        """
        super(FineTuneAlexNet, self).__init__()
        self.device = device
        self.patience = 3 # 정체로 간주할 연속 에폭 수
        self.weight_decay = 5e-4 # 가중치 감쇠 
        self.decay_factor = 0.1 # 학습률 감소 비율 (1/10)
        self.stage = 0 # 동결해제 단계
        self.lr = 1e-5 # 초기 학습률
        
        # 사전 훈련된 AlexNet 모델 불러오기
        self.alexnet = alexnet(weights=AlexNet_Weights.DEFAULT) 

        # 마지막 레이어를 데이터셋의 클래스 수에 맞게 변경
        self.alexnet.classifier[-1] = nn.Linear(4096, num_classes)

        # 모델 디바이스로 이동 - 옵티마이저는 생성될때의 파라미터 위치를 기억함
        self.alexnet.to(self.device)  

        # 손실 함수와 옵티마이저 정의
        class_weights = [1.0 / class_counts[i] for i in range(len(class_counts))] # 클래스 불균형 해결을 위한 가중치 계산
        weights_tensor = torch.FloatTensor(class_weights).to(device) # 텐서로 변환 및 디바이스로 이동        
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor) # 가중치가 적용된 손실 함수 정의

        # 1. 모든 파라미터를 먼저 동결
        for param in self.alexnet.parameters():
            param.requires_grad = False
            
        # 2. 새로 추가한 마지막 레이어의 파라미터만 동결해제해 학습하도록 설정
        for param in self.alexnet.classifier[-1].parameters():
            param.requires_grad = True

        # 옵티마이저 설정
        self._update_optimizer()

    def forward(self, x):
        return self.alexnet(x)
    
    def _update_optimizer(self):
        """
        옵티마이저 변경 함수
        - 현재 학습률과 가중치 감쇠를 사용해 Adam 옵티마이저 생성
        - 동결 해제된 파라미터만 옵티마이저에 전달
        """
        trainable_params = filter(lambda p: p.requires_grad, self.alexnet.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
    
    def detect_learning_plateau(self, e_metric):
        """
        학습 정체 감지 메서드
        - e_metric: 평가지표 기록 리스트 (Loss =  낮을수록좋음 F1,Acc = 높을수록좋음)
        """
        # patience + 1개 미만 판단불가
        if len(e_metric) < self.patience + 1: 
            return False
        
        # 최근 patience + 1개 지표가 모두 이전 것과 같거나 나쁜지 확인
        recent = e_metric[-(self.patience + 1):]

        # 기본값
        is_plateau = False

        for i in range(1, len(recent)):
            if recent[i] <= recent[i - 1]: 
                is_plateau = True
                break
        # # Loss 기준 정체 판단
        # for i in range(1, len(recent)):
        #     if recent[i] >= recent[i - 1]: 
        #         is_plateau = True
        #         break
        if is_plateau:
            logger.info("성능이 개선 되지 않아 모델이 저장되지 않았습니다.")
        return is_plateau
    
    def unfreeze_layers(self):
        """
        레이어를 동결 해제하는 메서드
        """
        self.stage += 1
        
        if self.stage == 1:
            for param in self.alexnet.classifier[4].parameters():
                param.requires_grad = True
            
            # 학습률 감소 적용
            self.lr = self.lr * self.decay_factor 

        elif self.stage == 2:
            for param in self.alexnet.classifier[1].parameters():
                param.requires_grad = True
            
            self.lr = self.lr * self.decay_factor 

        else:# 종료
            logger.info("모든 레이어가 동결 해제 되었습니다. 추가 동결 해제는 불가능합니다.")
            return False  

        # 변경된 파라미터와 학습률로 옵티마이저 재설정
        self._update_optimizer()
        return True # 계속 학습 진행

   
# 모델 저장 클래스 --------------------------------------------------------------
class SaveModel:
    """
    모델저장 클래스
    검증 손실 개선 시 최고 모델 저장,업데이트
    save : 에폭별 정보/모델 저장
    load : 저장된 모델 불러와 재사용

    """
    # def __init__(self, best_valid_loss=float('inf')): # Loss 기준
    def __init__(self, best_score=0.0): # F1 기준
        """
        생성자(초기화 함수)
        검증 손실 값 입력

        - best_valid_loss: 검증 손실 값 저장
        - best_score: 성능 지표 값 저장
        - AlexNet_Weights: 모델 지정
        - len(species_idx["data"]): 분류 클래스 수 계산해 지정
        """
        # self.best_valid_loss = best_valid_loss
        self.best_score = best_score
        self.model_class = alexnet
        self.model_args = ()        
        self.model_kwargs = {'weights': AlexNet_Weights.DEFAULT}
        self.num_classes = len(species_idx["data"]) 

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        """
        최고 성능 모델 저장 - 검증 손실 개선 시 모델 저장        
        - current_valid_loss: 현재 epoch의 검증 손실 값
        - epoch: 현재 학습 epoch 번호
        - model: 학습 중인 모델 객체
        - optimizer: 모델 최적화 도구 객체
        - criterion: 손실 함수 객체

        - 현재 검증 손실이 최고 성능보다 좋으면 저장 진행
        - DataParallel 적용 여부에 따라 모델 상태 사전 추출
        - 모델, 옵티마이저 상태, 손실 함수 정보 등을 best_model_path에 저장
        - 최고 성능 검증 손실 값 갱신 및 저장 완료 메시지 출력, 저장여부 출력
        """

        # 수정된 F1 기준 모델 저장
        # 성능 개선 안됨
        if current_valid_loss <= self.best_score:
            return False
        
        # 성능 개선됨 - 모델 저장
        self.best_score = current_valid_loss
        logger.info(f"\n모델 업데이트: {self.best_score}")
        logger.info(f"\n학습횟수: {epoch+1}\n")
    
        if hasattr(model, 'module'):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_name': type(criterion).__name__,
            'loss_params': criterion.__dict__
        }, f'{model_path}/AlexNet_model.pth')
        logger.info(f"모델 저장 성공. 경로 : {model_path}/AlexNet_model.pth")
        return True
    
        # # 기존 손실 기준 모델 저장
        # # 성능 개선 안됨
        # if current_valid_loss >= self.best_valid_loss:
        #     return False
        
        # # 성능 개선됨 - 모델 저장
        # self.best_valid_loss = current_valid_loss
        # logger.info(f"\nBest validation loss: {self.best_valid_loss}")
        # logger.info(f"\nSaving best model for epoch: {epoch+1}\n")
    
        # if hasattr(model, 'module'):
        #     model_state_dict = model.module.state_dict()
        # else:
        #     model_state_dict = model.state_dict()
        
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model_state_dict,
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss_name': type(criterion).__name__,
        #     'loss_params': criterion.__dict__
        # }, f'{model_path}/AlexNet_model.pth')
        # logger.info(f"모델 저장 성공. 경로 : {model_path}/AlexNet_model.pth")
        # return True


#---------------------
from torch.utils.data import DataLoader, Subset


batch_size = 32 # 32 
num_workers = 4 # 할당코어수, windows는 멀티프로세싱땜시 높이면 오히려 느려질수도 있다함
prefetch_factor= 4 # 각 워커가 미리 로드하는 배치 수 - rem
subset = None # 일부만 학습하고 싶을때
train_subset = None
test_subset = None


if subset is not None: # 일부만 학습
    """
    데이터가 너무 커서 일부만 테스트시키고 싶어서 만듦
    subset에 학습 시키고 싶은 데이터수를 넣으면 그만큼만 부분 집합 으로 학습시킴
    클래스와 상관없이 처음 부분부터 n개를 뽑아 학습시키기 때문에 테스트 용도로만 적합함 
    안돌아가니 에폭을 줄이시길
    """
    logger.info(f"subsetting data to {subset} results")
    train_subset_indices = list(range(subset if subset < len(train_dataset) else len(train_dataset)))
    train_subset = Subset(train_dataset, train_subset_indices)


    test_subset_indices = list(range(subset if subset < len(test_dataset) else len(test_dataset)))
    test_subset = Subset(test_dataset, test_subset_indices)


# DataLoader 생성 (데이터셋, 배치 크기, 셔플 여부, 워커 수 등 설정)
# 뭐지 이거????? 내가했을때는 작동 됬었는데???? 아 에폭수를 착각했나봄
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )




#----------------------------------------
# 훈련 및 검증 로직

# 모델 효율성 측정 함수
def model_info(model, device, input_size=(1, 3, 224, 224)):
    """
    모델 최적화 비교를 위한 정량적 지표(Params, FLOPs, FPS) 측정 및 출력
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)

    logger.info(f"{'-'*40}")

    # 1. 파라미터 및 연산량 (thop 이용)
    try:
        # thop.profile은 잡음(출력)이 좀 있어서 verbose=False 추천
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        flops = macs * 2 # 통상적으로 FLOPs는 MACs의 2배
        
        # 보기 좋게 포맷팅 (예: 10000 -> 10K)
        flops_str, params_str = clever_format([flops, params], "%.2f")
        logger.info(f"   - 크기           : {params_str}")
        logger.info(f"   - 연산량         : {flops_str}")
    except Exception as e:
        logger.info(f" thop 라이브러리 에러: {e}")

    # 2. 메모리 사용량 (근사치)
    total_params = sum(p.numel() for p in model.parameters())
    memory_mb = total_params * 4 / (1024 ** 2) # Float32 기준
    logger.info(f"   - 메모리 사용량  : {memory_mb:.2f} MB (Only Weights)")

    # 3. 추론 속도 (FPS)
    logger.info(f"추론 속도")
    
    # 웜업 (Warm-up)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 실제 측정 (반복 100회)
    iterations = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize() # GPU 시간 동기화 필수
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    fps = 1 / avg_time
    
    logger.info(f"   - 지연시간       : {avg_time*1000:.4f} ms")
    logger.info(f"   - 속도           : {fps:.2f} frames/sec")
    logger.info(f"{'-'*40}\n")

# SaveModel 인스턴스 생성
save_best_model = SaveModel()

# 검증지표
metric_collection = MetricCollection({
    'Pre': MulticlassPrecision(num_classes=num_classes, average='macro'), # MacroPre 라고 했다가 이상해서 그냥 그대로씀
    'F1': MulticlassF1Score(num_classes=num_classes, average='macro'),
    'Bal_Acc': MulticlassRecall(num_classes=num_classes, average='macro'),
    'Top5_Acc': MulticlassAccuracy(num_classes=num_classes, top_k=5)
}).to(device)



def train():
    """
    모델 훈련 및 검증 함수
    총 epochs 동안 모델 훈련 및 검증 수행
    """
    epochs = 50  # 총 에폭 수 설정 50 

    # 모델 생성
    model = FineTuneAlexNet(num_classes, class_counts, device)
    # 모델 효율성 측정
    model_info(model, device)
    
    train_metrics = metric_collection.clone() # 훈련용
    valid_metrics = metric_collection.clone() # 검증용

    train_pre = [] # 훈련 정밀도 기록 리스트
    valid_pre = [] # 검증 정밀도 기록 리스트
    train_f1s = [] # 훈련 F1 점수 기록 리스트
    valid_f1s = [] # 검증 F1 점수 기록 리스트
    bal_accs = [] # 검증 균형 정확도 기록 리스트
    top5_accs = [] # 검증 Top-5 정확도 기록 리스트
    train_losses = [] # 훈련 손실 기록 리스트
    valid_losses = [] # 검증 손실 기록 리스트

    logger.info("\n스크립트 초기 설정이 완료되었습니다. 훈련을 시작할 준비가 되었습니다.")
    logger.info("레이어 단계: 0")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch} of {epochs}")

        model_lr = model.optimizer.param_groups[0]['lr'] # 현재 학습률

        model.train()  # 모델을 훈련 모드로 설정
        gpu_augmentation.train() # 증강 모듈드 휸련 모드로
        train_running_loss = 0.0
        train_metrics.reset() # 에폭 시작시 훈련 지표 초기화
        # save_best_model = save_best_model(valid_loss, epoch, model, model.optimizer, model.criterion)


        # tqdm을 사용해 진행률 표시
        prog_bar = tqdm(train_loader, desc="Training", leave=False)
        for i, data in enumerate(prog_bar):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            images = gpu_augmentation(images) # gpu에서 블러등 처리 수행

            model.optimizer.zero_grad()  # 옵티마이저의 기울기 초기화
           
            outputs = model(images)  # 모델의 예측값 계산
            loss = model.criterion(outputs, labels)  # 손실 계산
           
            loss.backward()  # 역전파를 통해 기울기 계산
            model.optimizer.step()  # 옵티마이저를 통해 가중치 업데이트
            
            train_running_loss += loss.item()
            
            # 훈련 지표 측정기에 업데이트
            train_metrics.update(outputs, labels)
        #기존 손실계산
        train_loss = train_running_loss / len(train_loader)
        train_losses.append(train_loss)

     
        # 에폭 종료 후 최종 훈련 지표 계산 (딕형태 반환)
        train_results = train_metrics.compute()

        #추가 검증 계산
        train_f1s.append(train_results['F1'].item())  # 훈련 F1 점수 기록
        train_pre.append(train_results['Pre'].item()) # 훈련 정밀도 점수 기록

        train_metrics.reset()  # 훈련 지표 초기화
        
        # 모델 검증
        model.eval()  # 모델을 평가 모드로 설정
        gpu_normalization.eval()
        valid_metrics.reset()  # 검증 지표 초기화
        valid_running_loss = 0.0
       
        with torch.no_grad():  # 기울기 계산 비활성화
            prog_bar = tqdm(test_loader, desc="Validating", leave=False)
            for i, data in enumerate(prog_bar):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                images = gpu_normalization(images) #정규화만 수행

                model.outputs = model(images)
                loss = model.criterion(model.outputs, labels)
               
                valid_running_loss += loss.item()

                valid_metrics.update(model.outputs, labels) # 검증 지표 측정기에 업데이트
        #기존 손실계산 
        valid_loss = valid_running_loss / len(test_loader)
        valid_losses.append(valid_loss)

        # 에폭 종료 후 최종 검증 지표 계산 (딕형태 반환)
        valid_results = valid_metrics.compute()

        # 추가 검증 지표 기록
        valid_f1s.append(valid_results['F1'].item())
        bal_accs.append(valid_results['Bal_Acc'].item())
        top5_accs.append(valid_results['Top5_Acc'].item())
        valid_pre.append(valid_results['Pre'].item())
        
        logger.info(f"Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")
        logger.info(f"\n[Epoch {epoch+1}] Summary:")
        logger.info(f"Loss     | Train: {train_loss:.4f} | Valid: {valid_loss:.4f}")
        logger.info(f"F1-Score | Train: {train_results['F1']:.4f} | Valid: {valid_results['F1']:.4f}")
        logger.info(f"Bal_Acc  | Train: {train_results['Bal_Acc']:.4f} | Valid: {valid_results['Bal_Acc']:.4f}")
        logger.info(f"Top-5    | Train: {train_results['Top5_Acc']:.4f} | Valid: {valid_results['Top5_Acc']:.4f}")
        logger.info(f"Precis'n | Train: {train_results['Pre']:.4f} | Valid: {valid_results['Pre']:.4f}")
        logger.info("-" * 60)

        # 학습 정체 감지를 위해 저장시도
        save_model = save_best_model(valid_results['F1'].item(), epoch, model, model.optimizer, model.criterion)
        if save_model:
            logger.info("모델 성능 개선으로 모델이 저장되었습니다.")
        else:
            # 학습 정체 감지
            if model.detect_learning_plateau(valid_f1s):
                model_unfrozen = model.unfreeze_layers()
                # 레이어 동결 해제
                if model_unfrozen:
                    logger.info(f"레이어 동결해제 성공. 단계 : {model.stage}. 학습률이 {model.lr}로 감소되었습니다.")
                else:
                    # 모든 레이어가 동결 해제된 상태
                    logger.info("학습을 종료합니다.")
                    break  

    logger.info('훈련이 완료되었습니다.')

    # 훈련 과정의 손실 그래프 그리기
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, color='green', linestyle='-', label='train loss')
    plt.plot(valid_losses, color='blue', linestyle='-', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_path}models/AlexNet_loss.png')
    # plt.show()
    plt.close()

    # 평가 지표 그래프
    plt.figure(figsize=(10, 5))
    # F1 - train_f1s, valid_f1s
    plt.plot(train_f1s, color='lightcoral', linestyle='-', label='Train F1 (Macro)')
    plt.plot(valid_f1s, color='red', linestyle='--', label='Val F1 (Macro)')
    # Pre - pre
    plt.plot(train_pre, color='lightsteelblue', linestyle='-', label='Train Precision (Macro)')
    plt.plot(valid_pre, color='blue', linestyle='--', label='Val Precision (Macro)')
    # Bal_Acc - bal_accs
    plt.plot(bal_accs, color='black', linestyle='-', label='Balanced Acc')
    # Top5_Acc - top5_accs
    plt.plot(top5_accs, color='dimgrey', linestyle='--', label='Top-5 Acc')
    
    plt.title('Evaluation Metrics History (F1, Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Score (0.0 to 1.0)')
    plt.legend()
    # 저장
    plt.savefig(f'{output_path}models/AlexNet_evaluation_metrics.png') 
    # plt.show()
    plt.close()

    logger.info(f"손실 그래프가 {output_path}models/AlexNet_loss.png 에 저장되었습니다.")
    logger.info(f"평가 지표 그래프가 {output_path}models/AlexNet_evaluation_metrics.png 에 저장되었습니다.")

#=================================================================================
# 메인 실행
if __name__ == '__main__':
    import torch.multiprocessing # 멀티 프로세싱 충돌 방지
    torch.multiprocessing.freeze_support()
    train()



