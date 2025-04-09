# 응급 상황 인식 및 응급실 연계 서비스
- 사용자의 상황, 위치에 맞는 응급실 추천

![image](https://github.com/user-attachments/assets/4c4a1f6c-4c8c-41cc-be5f-282e60358d10)


---

**1️⃣** 음성 인식 및 요약

[음성 인식 및 요약 코드 바로가기](https://github.com/mingd00/Face-Recognition/blob/main/Notebooks/deepface_fer_model_test.ipynb)

- STT(Speech-to-Text): OpenAI의 **Whisper-1**
- 요약 및 핵심 키워드 도출: OpenAI의 **GPT-3.5-turbo**

---

**2️⃣**   응급 등급 분류

[응급 등급 분류 코드 바로가기](https://github.com/mingd00/ER-recommendation/blob/main/2.%20%EC%9D%91%EA%B8%89%20%EB%93%B1%EA%B8%89%20%EB%B6%84%EB%A5%98.ipynb)

- **BERT 모델**을 **파인 튜닝**하여 **응급 등급 분류**

---

**3️⃣** 응급실 추천

[응급실 연계(추천) 코드 바로가기](https://github.com/mingd00/ER-recommendation/blob/main/3.%20%EC%9D%91%EA%B8%89%EC%8B%A4%20%EC%97%B0%EA%B3%84(%EC%B6%94%EC%B2%9C).ipynb)

- **Naver Map API**를 이용해 **거리, 이동 시간 계산** 후 **가장 가까운 응급실 추천**

---

**4️⃣ 각 기능 모듈화 및 통합 파이프라인 생성**

[모듈화 코드 바로가기](https://github.com/mingd00/ER-recommendation/blob/main/4-1.%20%EB%AA%A8%EB%93%88%ED%99%94.ipynb)

[통합 파이프라인 생성 코드 바로가기](https://github.com/mingd00/ER-recommendation/blob/main/4-2.%20%ED%86%B5%ED%95%A9.ipynb)

- API 키 로딩
- 음성 파일 -> 텍스트 변환
- 텍스트 요약
- 모델 예측
- 거리 구하기
- 병원 추천

---

**5️⃣ Fast API로 Restful API 만들기**

[Fast API로 Restful API 생성 코드 바로가기](https://github.com/mingd00/ER-recommendation/blob/main/main.py)

- 상황, 위도, 경도 데이터를 보내면 응급 등급에 따라 가까운 병원을 반환

![image](https://github.com/user-attachments/assets/ba4bab1a-3db5-40b1-bcce-bb269bfa5628)


---

**패키지 설치**

```
pip install -r requirements.txt
```

---

**실행**

```
uvicorn main:app --reload
```

---

**기술스택**

- Python(Numpy, Pandas, Matplotlib)
- Tensorflow/Keras
- FastAPI
- Docker
- Azure

