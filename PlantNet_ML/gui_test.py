"""
파일명: gui_test.py
설명: 생서된 모델을 활용해 사용자가 직접 이미지를 선택하고, 학명 및 예측 확률을 확인할 수 있도록 구현된 데스크탑 GUI 응용 프로그램
- Tkinter 기반의 그래픽 인터페이스
"""


# 파일 열기
import PIL.Image
import PIL.ImageTk
import tkinter as tk
import tkinter.filedialog as fd
import torch
import torchvision.transforms as transforms
import pandas as pd
import plantnet_ML as plantnet
import json

# config
base_input_path = '' # 기본 입력경로 
input_path = f'{base_input_path}plantnet_300K/resized_images' # 이미지 
output_path = f'{base_input_path}output_data/' # 출력결과 
plantnet_metadata_path = f'{input_path}plantnet300K_metadata.json' # 메타 데이터 파일
species_idx_path = f'{input_path}class_idx_to_species_id.json'#종 id 파일
species_name_path = f'{input_path}plantnet300K_species_id_2_name.json'#학명 파일
model_path = f'{output_path}models/' # 모델 파일 경로
best_model_path = f'{model_path}best_model.pth' # 최고의 모델 파일 경로
convert_to_csv = True #csv변환여부

plantnet_class = plantnet.ModelManager()
model = plantnet_class.load_best_model(f'{best_model_path}')


metadata = pd.read_json(f'{output_path}metadata/metadata.json')
species_names = pd.read_csv(f'{output_path}metadata/species_names.csv')
species_names = dict(zip(species_names['species_idx'], species_names['species_name']))
 




def imageToData(filename): # 선택한 이미지 보이기, 이미지 전처리
"""
    이미지 파일 경로 입력받아 이미지 불러와 모델에 맞춰
    전처리된 텐서로 변환해 리턴
"""
    openImage = PIL.Image.open(filename).convert('RGB') #png 파일 오류 방지용 convert('RGB')

    dispImage = PIL.ImageTk.PhotoImage(openImage.resize((300,300))) # 화면 펴시용
    imageLabel.configure(image = dispImage)
    imageLabel.image = dispImage

    transform = transforms.Compose([
    transforms.Resize(256), # 이미지의 짧은 변 크기를 256으로 맞춤
    transforms.ToTensor(), # 텐서 변환 및 픽셀 값 정규화
    transforms.Normalize( # 채널별 정규화 
        mean=[0.5]*3,
        std=[0.5]*3
    ),
    transforms.CenterCrop(224) #중앙에서 224x224 자름
    ])
    tensorImage = transform(openImage).unsqueeze(0)
    return tensorImage


def predictDigits(data): # 학습된 모델을 로드.
"""
    입력받은 이미지 데이터를 로드된 모델에 넣고 식물 종 예측 및 결과 표시
"""
    with torch.no_grad():
        output = model(data)
        _, predicted = torch.max(output, 1) # 가장 높은 값의 인덱스 반환
        class_idx = predicted.item() # 예측된 클래스 인덱스        
        name = species_names.get(class_idx, "알수 없는 종") # 학명 가져오기

        # 확률 계산
        s_max = torch.softmax(output, 1) # 각 클래스에 대해 샘플이 속할 확률
        max_prob = torch.max(s_max).item() # 가장 높은 확률 반환
        prob = str(round(max_prob * 100, 2)) # 백분율로 변환 및 반올림

    textLabel.configure(text="이 식물의 id는"+str(class_idx)+"이고, 학명은"+str(name)+"일 확률이"+(prob)+"% 입니다.")

def openFile():
"""
    파일 탐색기로 이미지 선택 및 분류 겨로가 출력
"""
    fpath=fd.askopenfilename()
    if fpath:
        data=imageToData(fpath)
        predictDigits((data))


# ================== tkinter를 사용한 GUI ======================
root = tk.Tk()
root.title("식물 분류기") # 창 제목 추가
root.geometry("400x450") # 레이블이 길어질 수 있으므로 세로 길이 확보

btn = tk.Button(root, text="파일 열기", command=openFile)
btn.pack(pady=10)

imageLabel = tk.Label(root)
imageLabel.pack(pady=10)

textLabel = tk.Label(root, text="분류할 식물 사진을 선택하세요.")
textLabel.pack(pady=10)

root.mainloop()



