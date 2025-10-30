import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
# from PIL import Image # Pillow-SIMD 지원안함
import kornia.augmentation as K
import cv2
import os
import random
import torch # pythorch pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from tqdm import tqdm #실시간 진행 막대그래프
from collections import Counter

# config
base_input_path = 'C:/Users/gooaine/PlantNet/' # 아 이거 GPU문제 해결됨
input_path = f'{base_input_path}plantnet_300K/' # 데이터 폴더 "D:\ain2\PlantNet\plantnet_300K"
output_path = f'{base_input_path}output_data/' # 출력결과 "D:\ain2\PlantNet\output_data"
plantnet_metadata_path = f'{input_path}plantnet300K_metadata.json' # 메타 데이터 파일
images_path = f'{input_path}images_resized/'#종 id 파일
species_idx_path = f'{input_path}class_idx_to_species_id.json'#종 id 파일
species_name_path = f'{input_path}plantnet300K_species_id_2_name.json'#학명 파일
convert_to_csv = True #csv변환여부


metadata = pd.read_json(f'{output_path}metadata/metadata.json')
# species_idx = pd.read_json(f'{output_path}metadata/species_idx.json')
# species_idx = dict(zip(species_idx['species_idx'], species_idx['species_id']))

# 뭐지 난 딕형태로 친절하게 만들어줬는데 오히려 할일이 많아졌는걸

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

class ModelManager:
    # 원래 최고모델저장 클래스였지만 로드까지 맏게된 클래스
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.model_class = alexnet
        self.model_args = ()        
        self.model_kwargs = {'weights': AlexNet_Weights.DEFAULT}
        self.num_classes = len(species_idx["data"]) # 행길이가 1083이면서 2만 내놈 딕이라 다른가봄 이런 딕같은
       


    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
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
        }, best_model_path)
        print(f"모델 저장 성공. 경로 : {best_model_path}")

    def load_best_model(self,model_path):
        # 모델 객체를 내부에서 생성
        model = self.model_class(*self.model_args, **self.model_kwargs)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, self.num_classes)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def load_model_data(self, model):
        checkpoint = torch.load(best_model_path)
        # model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss_name = checkpoint['loss_name']
        loss_params = checkpoint['loss_params']
        return model, optimizer, epoch, loss_name, loss_params

    def save_model_data(self, epochs, model, optimizer, criterion):

        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_name': type(criterion).__name__,
            'loss_params': criterion.__dict__
        }, f'{output_path}models/model_data_{epochs}.pth')

    def save_model(self, epochs, model):

        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()  # DataParallel 상태
        else:
            state_dict = model.state_dict()         # 단일 모델 상태
        torch.save({'model_state_dict': state_dict}, f'{output_path}models/model_{epochs}.pth')







def show_species_sample(species_id):
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
    transforms.Resize((224, 224)), # AlexNet 입력 크기에 바로 맞춥니다.
    transforms.ToTensor()          # PIL 이미지를 PyTorch 텐서로 변환
])

# 데이터셋 정의
# transform에 클래스의 인스턴스를 전달. CPU 변환만 적용
train_dataset = datasets.ImageFolder(
    root=f'{images_path}train/',
    transform=simple_transform 
)

test_dataset = datasets.ImageFolder(
    root=f'{images_path}test/',
    transform=simple_transform 
)

#--------------------------------------------------------------

# 모델 정의 및 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 클래스별 이미지 수 계산
num_classes = len(species_idx["data"]) # 고유한 클래스의 수를 결정
# num_classes = len(train_dataset.classes) # 고유한 클래스의 수를 결정
class_counts = Counter(train_dataset.targets) # 각 클래스에 속하는 샘플의 수를 계산
# counts_per_class = [class_counts[i] for i in range(num_classes)] # 0부터 까지 각 클래스에 대한 개수 목록을 구성
# 클래스 불균형 해결을 위한 가중치 계산

# 손실 함수와 옵티마이저 정의
class_weights = [1.0 / class_counts[i] for i in range(len(class_counts))] # 클래스 불균형 해결을 위한 가중치 계산
weights_tensor = torch.FloatTensor(class_weights).to(device) # 텐서로 변환 및 디바이스로 이동

# 모델불러옴
model = alexnet(weights=AlexNet_Weights.DEFAULT) # 사전 훈련된 AlexNet 모델 불러오기

# 마지막 레이어만 교체
model.classifier[-1] = torch.nn.Linear(4096, num_classes) # 마지막 레이어를 데이터셋의 클래스 수에 맞게 변경


# CrossEntropyLoss에 가중치 적용
criterion = nn.CrossEntropyLoss(weight=weights_tensor) # 가중치가 적용된 손실 함수 정의

# for param in model.features.parameters():
#     param.requires_grad = False # 특징 추출기 부분의 파라미터를 고정

# 모든 파라미터를 먼저 동결
for param in model.parameters():
    param.requires_grad = False

# 동결해제, 새로 추가한 마지막 레이어의 파라미터만 학습하도록 설정
for param in model.classifier[-1].parameters():
    param.requires_grad = True

# # 특징 추출기(features) 부분은 동결
# for param in model.features.parameters():
#     param.requires_grad = False

# # 분류기(classifier) 부분은 모두 학습하도록 동결 해제
# for param in model.classifier.parameters():
#     param.requires_grad = True

model.to(device) # 모델을 디바이스로 이동

# optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam 옵티마이저 정의

# 동결이 해제된(학습이 필요한) 파라미터만 추려서 정의
params_to_update = filter(lambda p: p.requires_grad, model.parameters())
#optimizer = optim.Adam(params_to_update, lr=0.0001)

# 쓰읍 과적합 자꾸됨 학습률 낮추고 규제 강화함. 와 학습률을 대체 얼마나 낮추는거임
optimizer = optim.Adam(params_to_update, lr=1e-5, weight_decay=5e-4)

model = nn.DataParallel(model) # 다중 GPU 사용 설정
# ---

# ------------------------------------GPU가 담당할 증강 및 정규화 파이프라인 정의--
gpu_augmentation = nn.Sequential(
    # 사전 훈련 모델 사용시 반드시 해당 모델의 훈련에 사용됬던 것과 동일한 평균과 표준편차로 입력 이미지를 정규화 해야 한다.
    # 아니면 믿지못할 학습그래프가 나올것이다
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation(degrees=45.0, p=1.0),
    K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.5),
    # ImageNet 정규화
    K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
).to(device)

# 커스텀 로더 (BGR 형식 그대로 반환하도록 수정)
# def opencv_loader(path):
#     return cv2.imread(path)
gpu_normalization = nn.Sequential(
    K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
).to(device)

#---------------------
from torch.utils.data import DataLoader, Subset


batch_size = 32 # 32 
num_workers = 4 # 할당코어수, windows는 멀티프로세싱땜시 높이면 오히려 느려질수도 있다함
prefetch_factor= 4 # 각 워커가 미리 로드하는 배치 수 - rem
subset = None
train_subset = None
test_subset = None


if subset is not None:
    print(f"subsetting data to {subset} results")
    train_subset_indices = list(range(subset if subset < len(train_dataset) else len(train_dataset)))
    train_subset = Subset(train_dataset, train_subset_indices)


    test_subset_indices = list(range(subset if subset < len(test_dataset) else len(test_dataset)))
    test_subset = Subset(test_dataset, test_subset_indices)


# # DataLoader
# train_loader = DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True, 
#     num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor
# )
# test_loader = DataLoader(
#     test_dataset, batch_size=batch_size, shuffle=False, 
#     num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor
# )

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)




#----------------------------------------
# 훈련 및 검증 로직

print("\n스크립트 초기 설정이 완료되었습니다. 훈련을 시작할 준비가 되었습니다.")

# ModelManager 인스턴스 생성
save_best_model = ModelManager()


def train():
    epochs = 50  # 총 에폭 수 설정 50


    train_losses = [] #훈련 손실 - 각 학습 단계에(ephoch) 에서 발생하는 오차, 학습 데이터에 대한 예측과 실제 타깃 값 간의 차이
    valid_losses = [] #검증 손실 - 모델이 새로운 데이터에 대해 얼마나 잘 대응하는지에 대한 기록


    for epoch in range(epochs):
        print(f"Epoch {epoch} of {epochs}")
        # --- 모델 훈련(Training) ---

        model.train()  # 모델을 훈련 모드로 설정
        gpu_augmentation.train() # 증강 모듈드 휸련 모드로
        train_running_loss = 0.0
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
           
        train_loss = train_running_loss / len(train_loader)
        train_losses.append(train_loss)
       
        # --- 모델 검증(Validation) ---
        model.eval()  # 모델을 평가 모드로 설정
        gpu_normalization.eval()
        valid_running_loss = 0.0
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
               
        valid_loss = valid_running_loss / len(test_loader)
        valid_losses.append(valid_loss)
       
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")
       
        # 현재 에폭의 검증 손실을 기준으로 최고의 모델을 저장
        save_best_model(valid_loss, epoch, model, optimizer, criterion)
        # save_best_model.save_model(epoch, model, optimizer, criterion)
        # save_best_model.save_model_state(epoch, model)


    print('훈련이 완료되었습니다.')


    # 훈련 과정의 손실 그래프 그리기
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, color='green', linestyle='-', label='train loss')
    plt.plot(valid_losses, color='blue', linestyle='-', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_path}models/loss.png')
    plt.show()


    print(f"손실 그래프가 {output_path}models/loss.png 에 저장되었습니다.")


#=================================================================================
# 모델 저장


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    train()



