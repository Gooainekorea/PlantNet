# PlantNet_ML
Pl@ntNet-300K를 가지고 식물을 분류하는 머신러닝 학습 프로그램 입니다.

## 1. 환경설정

### 1. 데이터셋 다운
[Pl@ntNet-300K](https://zenodo.org/records/5645731) 에서 Version 1.1 다운해주세요
[메타데이터만 다운로드](https://lab.plantnet.org/seafile/d/bed81bc15e8944969cf6/)

### 2. 실행 경로
하드디스크에서 실행시 하드디스크의 입출력 속도에 맞춰집니다.
    -> HDD 100% CPU 30% GPU 1~3% 를 보게 됩니다.
처음부터 SSD(NVMe)에서 실행해 주세요

### 3. 가상환경 생성
1. 생성전 권한 주기
파워쉘 관리자 권한으로 열고 cd 가상환경 파일 경로
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser 인터넷에서 다운로드 된 스크립트는 서명 필요
Set-ExecutionPolicy Unrestricted -Scope CurrentUser 모든 스크립트 실행 허용 <- 난 이걸로함

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

## 파일 설명

