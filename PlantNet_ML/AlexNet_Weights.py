"""
파일명: AlexNet_Weights.py

AlexNet_Weights 모델의 모든 가중치 동결해 이미지 처리 부분 학습 특성을 유지,
최종 분류 레이어의 클래스 수에 맞게 재설계해 전이 학습 구현.

- Kornia 라이브러리를 활용한 GPU가속 데이터 증강을 구현해 효율을 높임
    학습을 위한 이미지 데이터 증강처리 - 뒤집기,회전,블러등 GPU에서 처리
    -> CPU 병목 현상을 크게 줄임

- 훈련 프로세스
    손실함수(class_weights) : 클래스 샘플 수에 반비례
    옵티마이저(Adam) : 학습률 감소, 가중치 감쇠
    DataLoader의 num_workers, pin_memory로 CPU에서 GPU로 데이터 전송 최적화

단순히 모델 학습 방향이 아닌 성능 향상을 위해
비교가 필요해 평가지수 측정 추가

torchmetrics 라이브러리 설치 (pip install torchmetrics, python -m pip install torchmetrics)

Macro Precision : 각 클래스 별로 예측값중 정답인 비율을 계산후 평균을 냄 average='macro'
Macro F1Score : Precision과 Recall의 조화평균
Balanced Accuracy : 클래스 정확도를 평균냄
Top-5 Accuracy : 다중 클래스에서 상위 5개 예측이 맞으면 정답으로 간주
Confusion Matrix (오차 행렬) : 학습 종료시 이미지로 저장

"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
# from PIL import Image # Pillow-SIMD 지원안함
import kornia.augmentation as K # GPU가속!!!!! 제발!!!! 아니 너무느려!!!!!!
import cv2
import os
import random
import torch # pythorch pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
import torch.nn as nn # 신경망 구성요소 및 모듈 제공
from torchvision.models import alexnet, AlexNet_Weights # 사전학습 모델 불러옴
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from tqdm import tqdm #실시간 진행 막대그래프
from collections import Counter
from torchmetrics import MetricCollection # ModelManager처럼 클래스 만들필요없음
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy # 다중분류 평가 지표

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

# 고집부리지말고 그냥 csv파일 불러다할껄
#--------------------------------------------------------------
plt.style.use('ggplot')

model_path = os.path.join(output_path, 'models')
os.makedirs(model_path, exist_ok=True)
best_model_path = os.path.join(model_path, 'best_model.pth')

class ModelManager:
    """
    원래 최고모델저장 클래스였지만 로드까지 맏게된 클래스
    
    그냥 쓸때 : 검증 손실 개선 시 최고 모델 저장,업데이트
    save : 에폭별 정보/모델 저장
    load : 저장된 모델 불러와 재사용

    """
    def __init__(self, best_valid_loss=float('inf')):
        """
        생성자(초기화 함수)
        검증 손실 값 입력

        검증 손실 값 저장 - best_valid_loss
        모델 지정 - AlexNet_Weights
        분류 클래스 수 계산해 지정 - len(species_idx["data"])
        """
        self.best_valid_loss = best_valid_loss
        self.model_class = alexnet
        self.model_args = ()        
        self.model_kwargs = {'weights': AlexNet_Weights.DEFAULT}
        self.num_classes = len(species_idx["data"]) 

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        """
        최그 성능 모델 저장 - 검증 손실 개선 시 모델 저장
        학습 도중 최고 성능 모델을 자동으로 저장하여 나중에 재사용할 수 있게 함
        
        - current_valid_loss: 현재 epoch의 검증 손실 값
        - epoch: 현재 학습 epoch 번호
        - model: 학습 중인 모델 객체
        - optimizer: 모델 최적화 도구 객체
        - criterion: 손실 함수 객체

        - 현재 검증 손실이 최고 성능보다 좋으면 저장 진행
        - DataParallel 적용 여부에 따라 모델 상태 사전 추출
        - 모델, 옵티마이저 상태, 손실 함수 정보 등을 best_model_path에 저장
        - 최고 성능 검증 손실 값 갱신 및 저장 완료 메시지 출력
        """
        if current_valid_loss >= self.best_valid_loss:
            return
        self.best_valid_loss = current_valid_loss
        print(f"\nBest validation loss: {self.best_valid_loss}")
        print(f"\nSaving best model for epoch: {epoch+1}\n")
    
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
        print(f"모델 저장 성공. 경로 : {model_path}/AlexNet_model.pth")

    def load_best_model(self,model_path):
        """
        저장된 최고 성능 모델 로드, 내부서 모델 객체 생성 - 바로쓰게함

        - model_path: 저장된 모델 파일 경로

        - 내부적으로 alexnet 모델 객체 생성 및 클래스 수에 맞게 출력 레이어 교체
        - 저장된 가중치 로드 및 평가 모드로 전환
        - 준비된 모델 객체 반환
        """
        model = self.model_class(*self.model_args, **self.model_kwargs)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, self.num_classes)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def load_model_data(self, model):
        """
        최고 성능 모델의 추가 학습에 필요한 학습 상태 정보 로드
        학습 중단 후 이어서 재학습 시 필요한 상태 정보 복원에 사용
        데이터 학습 중간점검 필요로 만든거라 삭제해도 무방

        - model: 미리 생성된 모델 객체

        - best_model_path에 저장된 체크포인트 불러옴
        - 옵티마이저 상태 복원
        - 학습 epoch, 손실 이름 및 파라미터 반환
        """
        checkpoint = torch.load(best_model_path)
        # model.load_state_dict(checkpoint['model_state_dict']) # 모델 상태
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss_name = checkpoint['loss_name']
        loss_params = checkpoint['loss_params']
        return model, optimizer, epoch, loss_name, loss_params

    def save_model_data(self, epochs, model, optimizer, criterion):
        """
        특정 epoch 시점의 모델과 학습 상태 저장를 epoch 넘버에 맞춘 파일명으로 저장
        학습의 특정 시점마다 상태를 저장해 두어 중간 재개 및 분석 가능
        역시 삭제해도 무방

        - epochs: 현재 epoch 번호 (int)
        - model: 학습 중인 모델 객체
        - optimizer: 최적화 도구 객체
        - criterion: 손실 함수 객체
        """
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_name': type(criterion).__name__,
            'loss_params': criterion.__dict__
        }, f'{output_path}models/model_data_{epochs}.pth')

    def save_model(self, epochs, model):
        """
        모델 가중치만 별도로 저장
        
        - epochs: 현재 epoch 번호
        - model: 저장할 모델 객체

        - 모델 가중치만 저장
        
        """

        state_dict = model.state_dict() 
        torch.save({'model_state_dict': state_dict}, f'{output_path}models/model_{epochs}.pth')

def show_species_sample(species_id): # 참고에 있었어
    # List all files in the directory
    directory_path = f"{images_path}train/{species_id}/"
    all_files = os.listdir(directory_path)


    # Filter out any non-image files if needed (e.g., based on file extension)
    image_files = [f for f in all_files if f.lower().endswith(('jpg'))]
    random_image_file = random.choice(image_files)
    image_path = os.path.join(directory_path, random_image_file)

    # Select a random image file
    random_image_file = random.choice(image_files)


    # Open and display the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

#----------------------------------------

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

#--------------------------------------------------------------

# 모델 정의 및 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 클래스별 이미지 수 계산
num_classes = len(species_idx["data"]) # 고유한 클래스의 수를 결정
class_counts = Counter(train_dataset.targets) # 각 클래스에 속하는 샘플의 수를 계산

# 손실 함수와 옵티마이저 정의
class_weights = [1.0 / class_counts[i] for i in range(len(class_counts))] # 클래스 불균형 해결을 위한 가중치 계산
weights_tensor = torch.FloatTensor(class_weights).to(device) # 텐서로 변환 및 디바이스로 이동

# 모델불러옴
model = alexnet(weights=AlexNet_Weights.DEFAULT) # 사전 훈련된 AlexNet 모델 불러오기

# 마지막 레이어만 교체
model.classifier[-1] = torch.nn.Linear(4096, num_classes) # 마지막 레이어를 데이터셋의 클래스 수에 맞게 변경


# CrossEntropyLoss에 가중치 적용
criterion = nn.CrossEntropyLoss(weight=weights_tensor) # 가중치가 적용된 손실 함수 정의

# 모든 파라미터를 먼저 동결
for param in model.parameters():
    param.requires_grad = False

# 동결해제, 새로 추가한 마지막 레이어의 파라미터만 학습하도록 설정
for param in model.classifier[-1].parameters():
    param.requires_grad = True


model.to(device) # 모델을 디바이스로 이동


# 동결이 해제된(학습이 필요한) 파라미터만 추려서 정의
params_to_update = filter(lambda p: p.requires_grad, model.parameters())

# 아 빨간약 먹었어

# Adam 옵티마이저 정의 - 학습률과 가중치 감쇠 설정
# e: 10의 거듭제곱. 1e-5 = 0.00001, 5e-4 = 0.0005
optimizer = optim.Adam(params_to_update, lr=1e-5, weight_decay=5e-4)



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
    print(f"subsetting data to {subset} results")
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

# ModelManager 인스턴스 생성
save_best_model = ModelManager()
metric_collection = MetricCollection({
    'Pre': MulticlassPrecision(num_classes=num_classes, average='macro'), # MacroPre 라고 했다가 이상해서 그냥 그대로씀
    'F1': MulticlassF1Score(num_classes=num_classes, average='macro'),
    'Bal_Acc': MulticlassRecall(num_classes=num_classes, average='macro'),
    'Top5_Acc': MulticlassAccuracy(num_classes=num_classes, top_k=5)
}).to(device)



def train():
    epochs = 50  # 총 에폭 수 설정 50 - 학습해봤더니 이정도로는 필요 없는듯
    """
    모델 훈련 및 검증 함수
    총 epochs 동안 모델 훈련 및 검증 수행
    """
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

    print("\n스크립트 초기 설정이 완료되었습니다. 훈련을 시작할 준비가 되었습니다.")

    for epoch in range(epochs):
        print(f"Epoch {epoch} of {epochs}")
        """
        --- 모델 훈련(Training) ---
        - 총 epochs 동안 반복 합습
        - 각 epoch마다 훈련/검증 손실 계산
        """
        model.train()  # 모델을 훈련 모드로 설정
        gpu_augmentation.train() # 증강 모듈드 휸련 모드로
        train_running_loss = 0.0

        train_metrics.reset() # 에폭 시작시 훈련 지표 초기화
        
        # tqdm을 사용해 진행률 표시
        prog_bar = tqdm(train_loader, desc="Training", leave=False)
        for i, data in enumerate(prog_bar):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            images = gpu_augmentation(images) # gpu에서 블러등 처리 수행

            optimizer.zero_grad()  # 옵티마이저의 기울기 초기화
           
            outputs = model(images)  # 모델의 예측값 계산
            loss = criterion(outputs, labels)  # 손실 계산
           
            loss.backward()  # 역전파를 통해 기울기 계산
            optimizer.step()  # 옵티마이저를 통해 가중치 업데이트
            
            train_running_loss += loss.item()
            
            # 훈련 지표 측정기에 업데이트
            train_metrics.update(outputs, labels)
        #기존 손실계산
        train_loss = train_running_loss / len(train_loader)
        train_losses.append(train_loss)
     
        # 에폭 종료 후 최종 훈련 지표 계산 (딕형태 반환)
        train_results = train_metrics.compute()

        #추가 검증 계산
        train_f1s.append(train_results['F1'].cpu().item())  # 훈련 F1 점수 기록
        train_pre.append(train_results['Pre'].cpu().item()) # 훈련 정밀도 점수 기록

        train_metrics.reset()  # 훈련 지표 초기화
        
        # --- 모델 검증(Validation) ---
        model.eval()  # 모델을 평가 모드로 설정
        gpu_normalization.eval()
        valid_running_loss = 0.0

        valid_metrics.reset()  # 검증 지표 초기화
        
        with torch.no_grad():  # 기울기 계산 비활성화
            prog_bar = tqdm(test_loader, desc="Validating", leave=False)
            for i, data in enumerate(prog_bar):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                images = gpu_normalization(images) #정규화만 수행

                outputs = model(images)
                loss = criterion(outputs, labels)
               
                valid_running_loss += loss.item()

                valid_metrics.update(outputs, labels) # 검증 지표 측정기에 업데이트
        #기존 손실계산 
        valid_loss = valid_running_loss / len(test_loader)
        valid_losses.append(valid_loss)

        # 에폭 종료 후 최종 검증 지표 계산 (딕형태 반환)
        valid_results = valid_metrics.compute()

        # 추가 검증 지표 기록
        valid_f1s.append(valid_results['F1'].cpu().item())
        bal_accs.append(valid_results['Bal_Acc'].cpu().item())
        top5_accs.append(valid_results['Top5_Acc'].cpu().item())
        valid_pre.append(valid_results['Pre'].cpu().item())
        
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")
        print(f"\n[Epoch {epoch+1}] Summary:")
        print(f"Loss     | Train: {train_loss:.4f} | Valid: {valid_loss:.4f}")
        print(f"F1-Score | Train: {train_results['F1']:.4f} | Valid: {valid_results['F1']:.4f}")
        print(f"Bal_Acc  | Train: {train_results['Bal_Acc']:.4f} | Valid: {valid_results['Bal_Acc']:.4f}")
        print(f"Top-5    | Train: {train_results['Top5_Acc']:.4f} | Valid: {valid_results['Top5_Acc']:.4f}")
        print(f"Precis'n | Train: {train_results['Pre']:.4f} | Valid: {valid_results['Pre']:.4f}")
        print("-" * 60)
        #현재 에폭의 검증 손실을 기준으로 최고의 모델을 저장
        save_best_model(valid_loss, epoch, model, optimizer, criterion)

    print('훈련이 완료되었습니다.')

    # 훈련 과정의 손실 그래프 그리기
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, color='green', linestyle='-', label='train loss')
    plt.plot(valid_losses, color='blue', linestyle='-', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_path}models/AlexNet_loss.png')
    plt.show()

    # 평가 지표 그래프
    plt.figure(figsize=(10, 5))
    # F1 - train_f1s, valid_f1s
    plt.plot(train_f1s, color='red', linestyle='-', label='Train F1 (Macro)')
    plt.plot(valid_f1s, color='darkred', linestyle='--', label='Val F1 (Macro)')
    # Pre - pre
    plt.plot(train_pre, color='blue', linestyle='-', label='Train Precision (Macro)')
    plt.plot(valid_pre, color='darkblue', linestyle='--', label='Val Precision (Macro)')
    # Bal_Acc - bal_accs
    plt.plot(bal_accs, color='orange', linestyle='-', label='Balanced Acc')
    # Top5_Acc - top5_accs
    plt.plot(top5_accs, color='purple', linestyle='--', label='Top-5 Acc')
    
    plt.title('Evaluation Metrics History (F1, Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Score (0.0 to 1.0)')
    plt.legend()
    # 저장
    plt.savefig(f'{output_path}models/AlexNet_evaluation_metrics.png') 
    plt.show()

    print(f"손실 그래프가 {output_path}models/AlexNet_loss.png 에 저장되었습니다.")
    print(f"평가 지표 그래프가 {output_path}models/AlexNet_evaluation_metrics.png 에 저장되었습니다.")
#=================================================================================
# 모델 저장
if __name__ == '__main__':
    import torch.multiprocessing # 멀티 프로세싱 충돌 방지
    torch.multiprocessing.freeze_support()
    train()


