# AGENTS.md

## Goal
이 프로젝트는 머신러닝 모델의 학습 결과를 기반으로
발표용 HTML 대시보드를 생성하는 것을 목표로 한다.

---

## Execution Steps

1. train.py를 실행하여 results.json 파일을 생성한다.
2. results.json을 읽어 모델 성능을 분석한다.
3. 분석 결과를 기반으로 dashboard.html을 생성한다.

---

## Commands

### 환경 실행
python train.py

---

## Expected Outputs

- results.json 생성
- dashboard.html 생성

---

## Dashboard Requirements

다음 요소를 반드시 포함해야 한다:

### 1. 모델 개요
- 모델: LogisticRegression + StandardScaler
- 데이터셋: Wine
- 전체 샘플 수 / feature 수

### 2. 데이터 분할
- Train size
- Test size

### 3. 교차검증 결과
- 평균 정확도 (mean)
- 표준편차 (std)
- 간단한 해석

### 4. 테스트 결과
- Accuracy
- classification report (precision, recall, f1-score)

### 5. 시각화
- 클래스별 성능 bar chart
- confusion matrix (또는 성능 요약 그래프)

### 6. 해석
- 모델 성능에 대한 자연어 설명

---

## Rules

- 반드시 results.json 기반으로 생성할 것
- HTML은 단일 파일로 생성
- 발표용으로 보기 좋게 구성
- 수치 + 시각화 + 해석 모두 포함

---

## Completion Criteria

다음 조건을 만족하면 작업 완료로 간주:

1. results.json이 생성되어 있음
2. dashboard.html이 생성되어 있음
3. 주요 성능 지표가 시각화되어 있음

---

## Failure Handling

- train.py 실행 실패 시 오류 메시지 확인 후 수정
- results.json이 없으면 반드시 먼저 생성
- 시각화가 깨지면 chart.js 포함하여 수정