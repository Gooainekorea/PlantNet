# 나도 할 수 있다! 머신러닝 학습 및 모델을 이용한 웹 어플리케이션 만들기
본 프로젝트는 크게 두 부분으로 나눠져 있습니다.

PlantNet_ML : 머신러닝 학습 프로그래밍을 구현

PlantNet_App : PlantNet_ML을 이용해 생성한 머신러닝 모델을 이용해 식물 분류 웹 어플리케이션을 구현


# 01. PlantNet_ML
예제코드 : [Olly Rennard님의 Kaggle - Pl@ntNet](https://www.kaggle.com/code/ollyrennard/pl-ntnet/notebook)

해당 머신러닝 학습 프로그래밍에서는 논문 결과의 재현 및 실현한 코드를 참고해 

어떻게 하면 더 효율적이고 안정적으로 모델을 학습 시킬 수 있을지에 초점을 맞췄습니다.

- 학습 파이프라인 최적화 및 실용적 구현
- kornia를 활용한 GPU가속 증강으로 CPU 병목 감소 (성능 및 효율성 향상)
- 클래스 불균형 해소를 위해 손실 함수에 가중치 적용
- 별도의 데이터 전처리 구현

자세한 부분은 [PlantNet_ML.md](PlantNet_ML/PlantNet_ML.md) 참고

# 02. PlantNet_App
PlantNet_ML을 이용해 생성한 머신러닝 모델을 이용해 식물 분류 웹 어플리케이션을 구현

FastAPI 를 이용해 파이썬 서버와 노드js서버 연동

웹js - 프론트엔드 동작

js서버  - 백엔드

파이썬서버 - 모델 돌리고 출력된 학명을 위키검색해 가져옴

---
# 단계별로 정리할 시간없어 올리는 포트폴리오 전문
● 프로젝트 개요
- PC 및 모바일(QR 코드 연동)을 통한 식물 사진 업로드
- PyTorch(AlexNet) 기반 딥러닝 모델을 이용한 식물 종 분류
- Wikipedia API 를 연동한 식물 정보(일반명, 설명) 조회 및 번역
- 분석된 식물 사진을 서버에 저장하는 기능

● 프로젝트 목표
- 사용자가 식물 사진을 업로드하면, 딥러닝 모델을 통해 식물의 종을 식별합니다.
- 식별된 식물의 학명, 일반명, 그리고 위키백과(Wikipedia) 기반의 상세 설명을 사용자에게 제공합니다.
- PC 와 모바일 환경 모두에서 손쉽게 이미지를 업로드하고 분석할 수 있는 직관적인 사용자 인터페이스(UI)를 구축합니다.
- 사용자가 업로드한 이미지를 서버에 저장하여 데이터베이스를 확장할 수 있는 기능을 제공합니다

● 테스트

에폭기록
진행별 학습그레프
CPU 병목 해결











● 딥러닝 워크플로우



● 최종 학습 그래프


● 계획

AI 모델 서버
실행 시나리오
계획
식물 분석하기 버튼누르면 app.js에서 식물사진 보내고 
server.js에서 사진이랑 요청 보내면 
plant_model.py에서 모델 불러다 분석해서 
server.js에 다시 주고 그걸 화면에 뿌려준다.

할것
plant_model.py:
PIL/tkinter GUI 관련 코드 제거
REST API 엔드포인트로 변환 (FastAPI 사용)
이미지를 받아서 모델로 분석하고 결과 반환하는 함수만 유지
server.js:
이미지 업로드를 처리하는 새로운 엔드포인트 추가
Python API로 이미지 전송하는 로직 추가
분석 결과를 클라이언트에 반환
app.js:
이미지 업로드 및 분석 버튼 클릭 시 서버로 이미지 전송
받은 분석 결과를 화면에 표시

확인사항
Python 웹 서버로 Flask나 FastAPI 중 어떤 것이 좋을지 
	Flask- 쉬움 FastAPI- 속도빠름, 디버깅효율좋음
	FastAPI
이미지 업로드 시 어떤 형식으로 전송할지 - multipart/form-data(이진데이터 형식)
서버 포트 설정
Express 서버의 포트 : 3000
Python 서버의 포트 : 5000

웹서버 : FastAPI
데이터전송 : multipart/form-data
포트 : Express 서버의 포트 : 3000, Python 서버의 포트 : 5000
1. Pl@ntNet-300K 데이터셋을 다운로드하고, 학습/검증/테스트 폴더로 정리.​
1. Python 3.8 이상 설치
2. Node.js LTS 버전 설치
3. (GPU 사용 시) NVIDIA 드라이버와 CUDA Toolkit 13.0을 설치
4. Node.js 서버에 필요한 패키지를 설치 npm install
5. Python 가상 환경을 만들고 활성화
      python -m venv venv -> venv\Scripts\activate
6. Python 서버에 필요한 패키지
      pip install –r src\requirements.txt
7. 방화벽 인바운드 규칙 설정
8. 두 개의 터미널을 열고 가상환경 활성화, 각각 서버를 실행
9. 터미널 1 (Node.js 서버): node server.js
10. 터미널 2 (Python 서버): uvicorn src.plant_model:app --host 127.0.0.1 --port 5000


너무 안되서 5시간 동안 깔고 지우고 하나씩 확인결과C++ 패키지가 손상됬던 거였다.
재설치후 문제없이 작동했다.


● 와이어프레임






● 파트별 상세 설명
Client Interface (Frontend) : js/app.js, index.html
사용자가 식물 이미지를 선택하고, 분석을 요청하며, 결과를 확인하는 전체 과정을 담당하는 싱글 페이지 애플리케이션
이미지 업로드:
1. PC : '사진 선택' 버튼을 통한 로컬 파일을 직접 업로드할 수 있습니다.
1. 모바일 QR 연동: qrcode.js 를 활용,
- 서버로부터 고유 세션 ID 와 로컬 IP 를 받아 QR 코드를 생성
- 사용자가 모바일로 QR 스캔 후 사진을 업로드,
- 프론트엔드는 서버의 상태를 주기적으로 확인하여 이미지가 준비되면 자동으로 화면에 표시.

API 통신: fetch API 를 사용하여 Node.js 서버의 엔드포인트(/api/analyze, /api/save-plant 등)와 비동기 통신을 수행
분석이 완료되면 서버로부터 받은 학명, 일반명, 설명 등의 데이터를 파싱하여 결과 창에
동적으로 렌더링

Application Layer (Backend) : server.js 3000
클라이언트와 AI 서비스(Python 서버) 간의 중재자(Mediator) 역할을 수행하는 Node.js 기반 API 서버.
API 라우팅: Express.js 를 사용하여 RESTful API 엔드포인트를 구축.
multer 미들웨어(node.js 파일업로드 미들웨어)를 사용하여 클라이언트로부터 업로드된 파일을 처리.
분석용 이미지는 메모리(buffer)에 저장하여 Python 서버로 전달 저장용 이미지는 디스크(images/plants/)에 영구 저장
QR 업로드 관리: 모바일 QR 업로드를 위해 세션 ID 를 키로 하는 임시 객체(mobileUploads)를 메모리에 유지. 모바일에서 사진이 업로드되면 해당 세션의 상태를 'completed'로 변경하고 이미지 데이터를 Base64 로 인코딩하여 저장
AI Service (Backend) : python_server.py 5000 FastAPI - 파이썬 서버 만듦
Postman - 파이썬 서버 응답 확인
역할: 실제 딥러닝 모델을 통해 이미지 분석 및 정보검색을 수행하는 Python 기반 서버. 
FastAPI -> PyTorch 모델(best_model.pth)을 로드, 엔드포인트를 통해 이미지 분석 요청을 처리.
입력된 이미지 바이트를 PIL 과 torchvision.transforms 를 통해 모델이 요구하는 텐서(Tensor) 형식으로 변환하고 정규화.
정보 검색 및 번역(get_prediction()):
1.모델이 예측한 식물의 학명을 기반으로 wikipediaapi 를 사용하여 영문 및 국문 위키백과 페이지를 검색.
2.국문 페이지가 없을 경우, 영문 요약 정보를 googletrans 라이브러리로 한국어로 번역하여 최종 설명과 일반명을 생성.
3.결과 반환: 분석된 학명, 일반명, 상세 설명을 포함한 JSON 객체를 Application Layer(Node.js 서버)로 반환.

● 실행 결과
