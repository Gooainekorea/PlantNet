"""
파일명: image_resizer.py
설명: cpu로 이미지 크기를 자르고 지정 경로에 저장.
"""
import os
import torch
import cv2 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
"""
너무 느려서 OpenCV로 처리 
pip install opencv-python
속도가 빨라졌다.
"""
# 경로 설정
base_input_path = ''
original_data_path = f'{base_input_path}plantnet_300K/images/'
preprocessed_data_path = f'{base_input_path}plantnet_300K/'
resized_images_path = os.path.join(preprocessed_data_path, 'resized_images')
os.makedirs(resized_images_path, exist_ok=True)

# 배치 사이즈 및 워커 설정 - cpu 병렬처리 최적화
BATCH_SIZE = 64
# 논리코어수
NUM_WORKERS = 6 


# --- 데이터셋 클래스 ---
class Image_Dataset(Dataset):
    def __init__(self, root_dir, target_dir, resize_dim=(256, 256)):
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.resize_dim = resize_dim # (w, h)
        self.image_paths = []
        self.target_paths = []

        if not os.path.exists(root_dir):
            return

        # 경로 없어서 저장 안됬음 돌겠네
        # 모든 경로를 리스트에 미리 추가
        for species_id in os.listdir(root_dir):
            source_species_dir = os.path.join(root_dir, species_id)
            if not os.path.isdir(source_species_dir):
                continue
            
            for image_name in os.listdir(source_species_dir):
                source_path = os.path.join(source_species_dir, image_name)
                target_path = os.path.join(target_dir, species_id, image_name)
                self.image_paths.append(source_path)
                self.target_paths.append(target_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # OpenCV로 이미지 읽기 (BGR 순서)
        image_bgr = cv2.imread(self.image_paths[idx])

        # 파일이 손상되었거나 읽을 수 없는 경우 처리
        if image_bgr is None:
            # 빈 이미지를 반환하여 나중에 건너뛸 수 있도록 함
            return np.zeros((self.resize_dim[1], self.resize_dim[0], 3), dtype=np.uint8), self.target_paths[idx], False

        # OpenCV로 리사이징 (INTER_AREA는 이미지 축소 시 품질이 좋음)
        resized_bgr = cv2.resize(image_bgr, self.resize_dim, interpolation=cv2.INTER_AREA)
        
        target_path = self.target_paths[idx]
            
        return resized_bgr, target_path, True


# --- 이미지 전처리 및 저장 함수 --- 
def save_images(source_dir, target_dir):
    # OpenCV를 사용하는 커스텀 데이터셋 생성
    dataset = Image_Dataset(source_dir, target_dir, resize_dim=(256, 256))
    
    # DataLoader는 NumPy 배열을 반환하도록 
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    for image_batch, path_batch, success_flags in tqdm(data_loader, desc=f'Processing {os.path.basename(source_dir)}'):
        for i in range(len(path_batch)):
            # 이미지를 성공적으로 읽었을 경우에만 저장
            if success_flags[i]:
                target_image_path = path_batch[i]
                os.makedirs(os.path.dirname(target_image_path), exist_ok=True)
                
                # image_batch[i]는 리사이징된 BGR 이미지 (NumPy 배열)
                # cv2.imwrite는 BGR 형식의 이미지를 그대로 저장하면 됨
                cv2.imwrite(target_image_path, image_batch[i].numpy())


# --- 메인 함수 ---
def main():
    print(f"Starting preprocessing with {NUM_WORKERS} workers.")
    for split in ['train', 'test']:
        source_split_dir = os.path.join(original_data_path, split)
        target_split_dir = os.path.join(resized_images_path, split)
        
        if os.path.exists(source_split_dir):
            # 이미지 전처리 및 저장을 실행
            save_images(source_split_dir, target_split_dir) # 안에서 객체생성

    print("Image preprocessing with OpenCV complete.")

#--- 메인

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()