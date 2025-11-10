# PlantNet_App
사용자가 업로드한 식물 사진을 학습된 모델을 통해 식물 종류를 판별하고 관련 정보를 제공하는 웹 애플리케이션 구현하기

- 앱디자인 : [v0.app](https://v0.app/)

## 기본 툴 설치
Python 3.8 이상 버전을 설치합니다.
Node.js LTS 버전을 설치합니다.
(GPU 사용 시) NVIDIA 드라이버와 CUDA Toolkit 13.0을 설치합니다.
터미널을 열고 PlantNet_app 폴더로 이동합니다.

Node.js 서버에 필요한 패키지를 설치합니다: npm install
Python 가상 환경을 만들고 활성화합니다: 

## 가상환경 생성
생성전 권한 주기 파워쉘 관리자 권한으로 열고 cd 가상환경 파일 경로
'Set-ExecutionPolicy RemoteSigned -Scope CurrentUser' 인터넷에서 다운로드 된 스크립트는 서명 필요

'Set-ExecutionPolicy Unrestricted -Scope CurrentUser' 모든 스크립트 실행 허용 <- 난 이걸로함

생성
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
파워셍 python -m pip install --upgrade pip

## 모델넣기
학습시킨모델을 src/model 안에 'best_model.pth'로 넣어줍니다 (python_server.py 22줄 참고)

src안에 metadata 폴더를 만들고 전처리한 메타데이터를 넣어줍니다.

## 패키지 설치
Python 서버에 필요한 패키지를 설치합니다: pip install -r src\requirements.txt
아... 잠깐 버전충돌 아 아아아아아ㅏㅏㅏ

두 개의 터미널을 열고 각각 서버를 실행합니다.
터미널 1 (Node.js 서버): node server.js
터미널 2 (Python 서버, 가상환경 활성화 필수): uvicorn src.python_server:app --host 127.0.0.1 --port 5000
    파워쉘 실행시 : python -m uvicorn src.python_server:app --host 127.0.0.1 --port 5000
