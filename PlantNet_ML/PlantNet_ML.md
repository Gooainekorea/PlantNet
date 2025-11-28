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

다운로드 후 프로젝트 루트 디렉토리의 'plantnet_300K/부


