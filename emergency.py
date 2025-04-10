import os
import requests
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import openai
from openai import OpenAI
import json
import torch


# 0. load key file------------------
def load_file(filepath):
    with open(filepath, 'r') as file:
        return file.readline().strip()


# 1-1 audio2text--------------------
def audio2text(audio_path, filename):
    client = OpenAI()
    audio_file = open(audio_path + filename, "rb")
    transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    language="ko",
                    response_format="text",
    )
    return transcript


# 1-2 text2summary------------------
def text2summary(input_text):
    # OpenAI 클라이언트 생성
    client = OpenAI()

    system_role = '당신은 입력된 텍스트를 가장 간결하고 핵심적인 표현으로 요약하는 어시스턴트입니다. 불필요한 설명을 생략하고 핵심적인 정보를 짧은 문장으로 요약하세요.'

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_role
            },
            {
                "role": "user",
                "content": f'"{input_text}" 어떤 상황인지 알 수 있게 문장자체를 요약해줘'
            }
        ]
    )

    return response.choices[0].message.content

# 2. model prediction------------------
def predict(text, model, tokenizer, device):
    # 입력 문장 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()} # 각 텐서를 GPU로 이동

    # 모델 예측
    with torch.no_grad():
        outputs = model(**inputs)

    # 로짓을 소프트맥스로 변환하여 확률 계산
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)

    # 가장 높은 확률을 가진 클래스 선택
    pred = torch.argmax(probabilities, dim=-1).item()

    return pred, probabilities



# 3-1. get_distance------------------
def get_dist(start_lat, start_lng, dest_lat, dest_lng, c_id, c_key):
    url = "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": c_id,
        "X-NCP-APIGW-API-KEY": c_key,
    }
    params = {
        "start": f"{start_lng},{start_lat}",  # 출발지 (경도, 위도)
        "goal": f"{dest_lng},{dest_lat}",    # 목적지 (경도, 위도)
        "option": "trafast"  # 실시간 빠른 길 옵션
    }

    # 요청하고 답변 받아오기 
    response = requests.get(url, headers=headers, params=params)
    response = response.json()

    if response['code'] == 1:
        dist = 0
    elif response['code'] == 0:
        # 'route' 키가 있는지 확인 후 접근
        if 'route' in response and 'trafast' in response['route']:
            dist = response['route']['trafast'][0]['summary']['distance']  # m(미터)
        else:
            print("No route found in the response.")
            dist = None  # 또는 기본값을 반환할 수 있음
    else:
        print(f"Unexpected response code: {response['code']}")
        dist = None

    return dist


# 3-2. recommendation------------------
def recommend_hospital(data, lat, lng, range, c_id, c_key) :
  # 특정 범위 데이터만 추출
  filtered_data = data[data['위도'].between(lat - range, lat + range) & data['경도'].between(lng - range, lng + range)].copy()

  filtered_data['거리'] = filtered_data.apply(lambda x: get_dist(lat, lng, x['위도'], x['경도'], c_id, c_key), axis=1)
  return filtered_data.sort_values(by='거리').head(3).reset_index(drop=True)
