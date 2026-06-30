# AI_EX_05_SVM
SVM 숫자 분류기

# 개요
OpenCV와 Scikit-learn을 사용하여 숫자 이미지를 학습하고, 달력 이미지에서 숫자를 예측하는 프로젝트입니다.
(사용자 정의 숫자 이미지 폴더(data/0 ~ data/9)를 이용한 학습)

예측 단계에서는 `digit.jpg` 이미지에서 숫자 영역을 추출한 뒤, 학습된 SVM 모델을 사용해 숫자를 분류합니다.

## Features

- 숫자 이미지 전처리
  - Grayscale 변환
  - Gaussian Blur
  - Otsu Threshold
  - Contour 기반 숫자 영역 추출
  - 28x28 크기 정규화
- SVM(`sklearn.svm.SVC`) 기반 숫자 분류
- 사용자 데이터셋 학습 지원(data/0 ~ data/9) -> kiggle에서 download 함
- `digit.jpg` 이미지 숫자 예측
- 예측 결과를 이미지로 저장

# 실행
  1) Train_SVM.py
     -> 숫자 학습
     -> 모델 생성 : svm_digit_model.pkl 
  2) Pred_SVM.py
     -> model load : svm_digit_model.pkl
     -> image 읽기 : digit.jpg
     -> 결과값 저장 : digit_result.jpg
