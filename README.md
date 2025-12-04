# K-Game-Stock-Prediction
2025년 빅데이터 수업시간에 진행한 대한민국 대표 게임사들의 재무상태 및 주가를 파악해보고, 굵직한 이벤트들과 연관지어보았습니다.

import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd

# API 인증키 입력
crtfc_key = 'be1cf4dc3136fbd49a414efc1dd8141b30c6354f'
url = f'https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={crtfc_key}'

# API 요청 및 ZIP 파일 받기
response = requests.get(url)
zip_bytes = io.BytesIO(response.content)

# ZIP 압축 해제
with zipfile.ZipFile(zip_bytes) as z:
    xml_filename = z.namelist()[0]
    with z.open(xml_filename) as xml_file:
        xml_content = xml_file.read()

# XML 파싱
root = ET.fromstring(xml_content)

# 기업코드 리스트 추출
corp_list = []
empty_list =[]
for corp in root.findall('list'):
  c_name = corp.findtext('corp_name')
  if c_name=='넥슨게임즈' or c_name=='엔씨소프트' or c_name=='크래프톤' or c_name=='카카오게임즈' or c_name=='넷마블':
    corp_list.append({
        'corp_code': corp.findtext('corp_code'),
        'corp_name': c_name,
        'stock_code': corp.findtext('stock_code'),
        'modify_date': corp.findtext('modify_date')
    })

# DataFrame으로 변환
corp_target_df = pd.DataFrame(corp_list)
print(corp_target_df.head())
