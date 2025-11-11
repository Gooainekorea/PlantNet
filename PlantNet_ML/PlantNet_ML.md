# PlantNet_ML
Pl@ntNet-300K를 가지고 식물을 분류하는 머신러닝 학습 프로그램 입니다.

## 구조
```
PlantNet_ML
├───metadata_tools.py                 # 데이터셋의 정보(메타데이터)를 가공하고 처리
├───image_resizer.py                  # 이미지 전처리를 위한 크기 조정 도구
├───plantnet.py                       # PyTorch 모델을 불러오고, 예측을 수행하는 등 머신러닝 핵심 로직을 담은 파일
├───gui_test.py                       # 시각적요소를 이용해 모델을 테스트할 수 있는 파일
├───requirements.txt                  # 프로젝트 실행에 필요한 파이썬 라이브러리 목록 파일
│
├───output_data\                      # 모델 학습 결과물 및 가공된 데이터가 저장되는 폴더
│   ├───models\                       # 생성된 모델 및 관련 파일 저장 폴더
│   └───metadata\                     # 위 원본 JSON 파일들을 사용하기 쉽게 CSV, JSON 등으로 변환하여 저장한 폴더
│
├───plantnet_300K\                    # 모델 학습 결과물 및 가공된 데이터가 저장되는 폴더
│   ├───images\                       # 이미지 파일 폴더
│   ├───resized_images\               # 크기조정된 이미지 파일 폴더
│   ├───class_idx_to_species_id.json  # 클래스 id(index)와 식물 종 ID를 연결 메타데이터 파일
│   ├───plantnet300K_metadata.json    # 각 이미지의 ID를 여러 정보(종 ID, 분할, 저자, 라이선스 등)와 매핑 메타데이터 파일
│   └───plantnet300K_species_id_2_name.json # 식물 종 ID와 실제 학명을 연결 메타데이터 파일

```

# 1. 환경설정

### 1. 데이터셋 다운
[Pl@ntNet-300K](https://zenodo.org/records/5645731) 에서 Version 1.1 다운해주세요

메타데이터만 다운받고 싶다면 : 

[메타데이터 다운](https://lab.plantnet.org/seafile/d/bed81bc15e8944969cf6/)

다운로드 후 프로젝트 루트 디렉토리의 'plantnet_300K/'에 압축 해제

### 2. 실행 경로
하드디스크에서 실행시 하드디스크의 입출력 속도에 맞춰집니다.

(습관적으로 하던 경로에 만들고 실행하다 HDD 100% CPU 30% GPU 1~3% 를 봄)

처음부터 SSD(NVMe)에서 실행해 주세요

### 3. 가상환경 생성

1. 생성전 권한 주기
파워쉘 관리자 권한으로 열고 cd 가상환경 파일 경로

'Set-ExecutionPolicy RemoteSigned -Scope CurrentUser' 인터넷에서 다운로드 된 스크립트는 서명 필요

'Set-ExecutionPolicy Unrestricted -Scope CurrentUser' 모든 스크립트 실행 허용 <- 난 이걸로함

2. 생성
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```

### 4. 환경체크
1. 장치관리자 - 디스플레이 어댑터 : 그래픽카드 종류 확인
[엔비디아 그래픽 카드 드라이버](https://www.nvidia.com/ko-kr/drivers/)

2. [해당 드라이버에 맞는 쿠다 둘킷 설치](https://developer.nvidia.com/cuda-downloads)
3. [PyTorch 설치](https://pytorch.org/get-started/locally/)
4. 위와 호환되는 파이썬 버전 설치

Python 버전 확인: python --version

PyTorch 버전 확인: import torch; print(torch.__version__)

torchvision 버전 확인: import torchvision; print(torchvision.__version__)

CUDA 버전 확인: nvidia-smi

# 파일 실행 순서
※ 파일상단의 경로부분 직접 수정 필요

1. 'matadata_tool.py'   : 메타데이터 전처리 - 인덱싱 파일 생성
2. 'image_resizer'      : 이미지 크기 처리 - 이미지 파일 생성
3. 'plantnet_ML.py'     : 머신러닝 데이터 학습
        ModelManager - 모델 저장/로드 담당 클래스
4. 'gui_test.py'        : 모델 테스트

# 딥러닝 워크플로우
![deep_learning_workflow](Gooainekorea/PlantNet/00_과정정리/img/deep_learning_workflow.png)

# 학습 곡선
![loss](Gooainekorea/PlantNet/00_과정정리/img/loss.png)


# 주요 기술 구현 포인트

