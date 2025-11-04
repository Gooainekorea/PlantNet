# 나도 할 수 있다! 머신러닝 학습 및 모델을 이용한 웹 어플리케이션 만들기
본 프로젝트는 크게 두 부분으로 나눠져 있습니다.

PlantNet_ML : 머신러닝 학습 프로그래밍을 구현

PlantNet_App : PlantNet_ML을 이용해 생성한 머신러닝 모델을 이용해 식물 분류 웹 어플리케이션을 구현


# PlantNet_ML
예제코드 : [Olly Rennard님의 Kaggle - Pl@ntNet](https://www.kaggle.com/code/ollyrennard/pl-ntnet/notebook)
(Olly님 감사합니다.)

해당 머신러닝 학습 프로그래밍에서는 논문 결과의 재현 및 실현한 코드를 참고해 

어떻게 하면 더 효율적이고 안정적으로 모델을 학습 시킬 수 있을지에 초점을 맞췄습니다.

- 학습 파이프라인 최적화 및 실용적 구현
- kornia를 활용한 GPU가속 증강으로 CPU 병목 감소 (성능 및 효율성 향상)
- 클래스 불균형 해소를 위해 손실 함수에 가중치 적용
- 별도의 데이터 전처리 구현

자세한 부분은 [PlantNet_ML.md](/PlantNet_ML.md) 참고

# PlantNet_App
