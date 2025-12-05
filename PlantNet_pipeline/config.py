"""
경로와 하이퍼파라미터 설정.
모델추가시 하이퍼파라미터 추가
"""
import os
import torch
import logging

class PathConfig:
    """
    파일 시스템 경로와 디렉토리 생성
    """
    def __init__(self, base_path='C:/PlantNet/'):
        self.base_path = base_path
        
        # 데이터 root
        self.data_root = os.path.join(base_path, 'data/')

        # 원본 데이터 경로 
        self.raw_data = os.path.join(self.data_root, 'plantnet_300K/')
        self.raw_images = os.path.join(self.raw_data, 'raw_images/')
        self.raw_metadata = os.path.join(self.raw_data, 'raw_metadata/')
        
        # 전처리된 데이터 경로 
        self.resized_images = os.path.join(self.data_root, 'resized_images/')
        self.processed_metadata = os.path.join(self.data_root, 'processed_metadata/')
        
        # 출력 및 저장 경로 (Outputs & Artifacts)
        self.results = os.path.join(base_path, 'results/')
        self.models = os.path.join(self.results, 'models/')
        self.checkpoints = os.path.join(self.results, 'checkpoints/')
        self.logs = os.path.join(self.results, 'logs/')

        # 초기화 시 디렉토리 자동 생성
        self._create_directories()

    def _create_directories(self):
        """
        디렉토리생성
        """
        dirs_to_create = [
            self.results,
            self.resized_images,
            self.processed_metadata,
            self.data_root, # in 압축파일 풀은 데이터셋
            self.models,
            self.checkpoints,
            self.logs
        ]
        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)
        try:
            logger = logging.getLogger('PathConfig')
            logger.info(f"디렉토리 생성 {self.base_path}")
            print(f"디랙토리 생성 {self.base_path}")
        except Exception as e:
            print(f"디랙토리 생성 실패: {e}")


class TrainConfig:
    """
    모델 학습에 필요한 하이퍼파라미터 25.12.05 yaml으로 빼기
    """
    def __init__(self, model_name='alexnet'):
        self.model_name = model_name.lower()
        
        # 하드웨어 설정 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 학습 루프 설정
        self.num_workers = 4
        self.batch_size = 32
        self.epochs = 50
        self.num_classes = 0
        
        # 모델별 정의
        self.model_params = {
            'alexnet': { # AlexNet 설정
                'resize_dim': (256, 256),
                'input_dim': (224, 224),
                'batch_size': 32,
                'learning_rate': 1e-5
            }
        }

        if self.model_name in self.model_params:
            params = self.model_params[self.model_name]
        else:
            raise ValueError(f"{self.model_name}모델은 지원하지 않습니다."
                             f"지원 모델: {list(self.model_params.keys())}")
        
    
if __name__ == "__main__":
    path_config = PathConfig(base_path='./test_project/')
    train_config = TrainConfig(model_name='alexnet')
    print("학습 장치:", train_config.device)