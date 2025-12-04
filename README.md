# K-Game-Stock-Prediction
2025년 빅데이터 수업시간에 진행한 대한민국 대표 게임사들의 재무상태 및 주가를 파악해보고, 굵직한 이벤트들과 연관지어보았습니다.

# 1. 기업 고유 코드(Corp Code) 추출
### Open Dart API를 호출하여 주요 게임사 5곳(크래프톤, 넥슨게임즈, 카카오게임즈, 엔씨소프트, 넷마블)의 고유 식별 코드를 확보합니다.
```
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
```

# 2. 공시 보고서 목록(Report List) 수집
### 확보한 고유 코드를 활용하여 2018년부터 2025년까지 발행된 모든 정기 공시(사업보고서, 분기/반기보고서)의 접수 번호와 원문 링크를 수집합니.
```
import requests

finance_filepath = None
stock_filepath = None

page_count = max

webpages = []

bsns_year = ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']

# 기업별 공시 검색 및 데이터 수집
for idx in corp_target_df.index:
    corp_code = corp_target_df.loc[idx, 'corp_code']
    corp_name = corp_target_df.loc[idx, 'corp_name']

    # 연도별 검색 기간 설정 및 API URL 생성
    for year in bsns_year:
        bgn_de = year + '0501'  # 1분기 보고서 발행월
        end_de = str(int(year) + 1) + '0430'  # 다음 해 3월까지

        url = (
            f'https://opendart.fss.or.kr/api/list.xml?crtfc_key={crtfc_key}'
            f'&corp_code={corp_code}&pblntf_ty=A'
            f'&bgn_de={bgn_de}&end_de={end_de}'
            f'&page_count=100'
            f'&pblntf_detail_ty=A001&pblntf_detail_ty=A002&pblntf_detail_ty=A003'
        )

        # API 요청 및 응답 디코딩
        resp = requests.get(url)
        content = resp.content.decode('UTF-8')

        # 수집 데이터 리스트에 추가
        webpages.append({
            'corp_name': corp_name,
            'year': year,
            'content': content
        })

# 수집 결과 확인
for page in webpages:
    print(f"{page['corp_name']} - {page['year']}")
    print(page['content'][:500])  # 앞부분만 출력
```

```
import xml.etree.ElementTree as ET
import pandas as pd

# 결과 저장 리스트 초기화
corp_records = []

# 수집된 페이지 순회 및 데이터 타입별 XML 추출
for webpage in webpages:
    if isinstance(webpage, dict):
        xml_text = webpage.get('content', '')
        src_corp_name = webpage.get('corp_name', None)
        src_year = webpage.get('year', None)
    elif isinstance(webpage, bytes):
        xml_text = webpage.decode('UTF-8')
        src_corp_name, src_year = None, None
    else:
        xml_text = str(webpage)  # 문자열로 강제 변환
        src_corp_name, src_year = None, None

    # XML 파싱 및 list 태그 탐색
    root_element = ET.fromstring(xml_text)
    iter_element = root_element.iter(tag="list")

    for element in iter_element:
        # 개별 공시 정보 및 기업 상세 정보 추출
        report_name = element.findtext("report_nm")
        corp_name = element.findtext("corp_name")
        corp_code = element.findtext("corp_code")
        stock_code = element.findtext("stock_code")
        corp_class = element.findtext("corp_cls")
        rcept_dt = element.findtext("rcept_dt")

        # 누락된 종목코드(stock_code) 보정
        if stock_code == " " or stock_code is None:
            match = corp_target_df[(corp_target_df['corp_name'] == corp_name) & (corp_target_df['stock_code'] != " ")]
            if not match.empty:
                stock_code = match['stock_code'].iloc[0]

        # 보고서 명칭 및 발행 연월 분리
        rept_nm, rept_dt = report_name.split(' ')
        rept_dt = rept_dt[1:-1]  # 괄호 제거
        report_date = rept_dt.split('.')  # ['YYYY', 'MM']

        # 보고서 종류별 고유 코드 할당
        if rept_nm == '반기보고서':
            report_code = '11012'  # 반기 보고서 코드
        elif rept_nm == '사업보고서':
            report_code = '11011'  # 사업 보고서 코드
        else:
            if report_date[1] == '03':
                report_code = '11013'  # 1분기 보고서 코드
            else:
                report_code = '11014'  # 3분기 보고서 코드

        # 정제된 데이터를 리스트에 추가
        corp_records.append({
            'corp_name': corp_name,
            'corp_code': corp_code,
            'stock_code': stock_code,
            'corp_class': corp_class,
            'report_nm': rept_nm,
            'report_date': '-'.join(report_date),
            'report_code': report_code,
            'src_year': src_year,
            'rcept_dt' : rcept_dt
        })

# DataFrame 변환 및 정렬
disclosures_df = pd.DataFrame(corp_records)
disclosures_df = disclosures_df.sort_values(by=['corp_name', 'report_date']).reset_index(drop=True)

# 결과 상위 20개 출력
print(disclosures_df.head(20))
```

```
import requests

bsns_year = ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']

requested_reports = []

for idx, row in disclosures_df.iterrows():
    corp_name = row['corp_name']
    corp_code = row['corp_code']
    report_date = row['report_date']   # YYYY-MM
    report_code = row['report_code']   # 11011, 11012, 11013, 11014 등
    year = report_date.split('-')[0]   # 보고서 연도
    rcept_dt = row['rcept_dt'] #접수 일자
    # 요청 URL 구성
    url = (
        f'https://opendart.fss.or.kr/api/fnlttSinglAcnt.xml?crtfc_key={crtfc_key}'
        f'&corp_code={corp_code}&bsns_year={year}&reprt_code={report_code}'
    )

    resp = requests.get(url)
    content = resp.content.decode('UTF-8')

    # 회사명, 연도, 보고서 코드와 함께 저장
    requested_reports.append({
        'corp_name': corp_name,
        'corp_code': corp_code,
        'year': year,
        'report_code': report_code,
        'content': content,
        'rcept_dt' : rcept_dt
    })

# 확인
for report in requested_reports[:5]:
    print(report['corp_name'], report['year'], report['report_code'])
    print(report['content'][:300])
```
# 3. 재무 데이터 파싱 및 추출
### XML 형태로 된 보고서 원문에서 핵심 재무 지표들을 뽑아냅니다.
```
import xml.etree.ElementTree as ET
import pandas as pd

# 결과 저장 리스트 초기화
all_report_data = []

# 금액 데이터 전처리 함수 정의
def clean_and_convert_amount(amount_str):
    """문자열 금액을 정수형으로 변환합니다."""
    try:
        # 금액에 콤마가 포함되어 있을 수 있으므로 제거 후 변환
        return int(amount_str.replace(',', '')) if amount_str else 0
    except:
        return 0

# 수집된 보고서 XML 순회 및 데이터 추출
for requested_report in requested_reports:
    xml_text = requested_report['content']
    corp_name = requested_report['corp_name']
    corp_code = requested_report['corp_code']
    year = requested_report['year']
    report_code = requested_report['report_code']
    report_rcept_dt = requested_report['rcept_dt']

    root_element = ET.fromstring(xml_text)

    # 계정별 상세 데이터 파싱 및 리스트 저장
    for item in root_element.findall("list"):
        data = {
            'corp_name': corp_name,
            'corp_code': corp_code,
            'bsns_year': item.findtext('bsns_year'),
            'reprt_code': item.findtext('reprt_code'),
            'account_nm': item.findtext('account_nm'),
            'fs_div': item.findtext('fs_div'),
            'sj_div': item.findtext('sj_div'),
            'thstrm_nm': item.findtext('thstrm_nm'),
            'thstrm_dt': item.findtext('thstrm_dt'),
            'thstrm_amount': item.findtext('thstrm_amount'),
            'frmtrm_nm': item.findtext('frmtrm_nm'),
            'frmtrm_dt': item.findtext('frmtrm_dt'),
            'frmtrm_amount': item.findtext('frmtrm_amount'),
            'currency': item.findtext('currency'),
            'rcept_no': item.findtext('rcept_no'),
            'rcept_dt' : report_rcept_dt,
        }

        # 당기 금액 정수형 변환 및 추가
        thstrm_amount_int = clean_and_convert_amount(data['thstrm_amount'])
        data['thstrm_amount_int'] = thstrm_amount_int

        # 대상 연도 데이터 필터링
        if data['bsns_year'] in bsns_year:
            all_report_data.append(data)

# DataFrame 변환
finance_df = pd.DataFrame(all_report_data)

# 숫자형 컬럼 변환 및 결측치 처리
finance_df['thstrm_amount_int'] = pd.to_numeric(
    finance_df['thstrm_amount'].astype(str).str.replace(',', ''),
    errors='coerce'
).fillna(0).astype(int)

# 데이터 정렬
finance_df = finance_df.sort_values(by=['corp_name', 'bsns_year', 'reprt_code']).reset_index(drop=True)

# 중복 데이터 제거
finance_df = finance_df.drop_duplicates(
    subset=['corp_name', 'bsns_year', 'reprt_code', 'account_nm'],
    keep='first'
)

# 결과 확인
print(finance_df)
```

# 4. 데이터 전처리
### 단순 누적치로 기재된 연말 사업보고서 데이터에서 1~3분기 실적을 역산하여 순수 4분기 실적을 도출하고, 결측치를 보간하여 분석 가능한 시계열 데이터셋 완성합니다.
```
import pandas as pd

need_account = [
    '매출액',
    '영업이익',
    '당기순이익(손실)',
    '자산총계',
    '부채총계',
    '자본총계',
    '자본금',
    '이익잉여금',
    '법인세차감전 순이익'
]

def year_quarter_classifier(code):
    if code == '11013':
        return 'Q1'
    elif code == '11012':
        return 'Q2'
    elif code == '11014':
        return 'Q3'
    elif code == '11011':
        return 'Q4'
    else:
        return None

def calculate_q4_solo_amount(df, target_account_nm):
    # 4분기 단독 계산 대상 행을 마스킹
    target_mask = (df['reprt_code'] == '11011') & (df['account_nm'] == target_account_nm)

    # 4분기 대상이 되는 행들만 복사
    df_q4 = df[target_mask].copy()

    if df_q4.empty:
        return df

    # 계산에 필요한 분기 코드 (1, 2, 3분기)
    quarter_codes = ['11014', '11012', '11013'] # Q3, Q2, Q1

    # 1, 2, 3분기 금액 합산 함수
    def get_q123_sum(row):
        corp_name = row['corp_name']
        bsns_year = row['bsns_year']

        sum_quarterly_amounts = 0

        # 1, 2, 3분기 데이터를 찾아서 합산
        for code in quarter_codes:
            prev_data = df[
                (df['corp_name'] == corp_name) &
                (df['bsns_year'] == bsns_year) &
                (df['account_nm'] == target_account_nm) &
                (df['reprt_code'] == code)
            ]

            if not prev_data.empty:
                # 'thstrm_amount' 컬럼 (이미 숫자형으로 전처리됨)을 사용하여 금액을 가져옴
                prev_amount = prev_data['thstrm_amount'].iloc[0]
                sum_quarterly_amounts += prev_amount
        return sum_quarterly_amounts

    # 4분기 연간 누적 금액에서 1, 2, 3분기 합계를 뺀다.
    q123_sum = df_q4.apply(get_q123_sum, axis=1)

    # 4분기 단독 금액 계산
    q4_solo_amount = df_q4['thstrm_amount'] - q123_sum

    # 계산된 4분기 단독 금액을 기존 df에 업데이트
    df.loc[target_mask, 'thstrm_amount'] = q4_solo_amount

    return df


df = finance_df[['corp_name','bsns_year','reprt_code','account_nm','thstrm_amount','frmtrm_amount','currency', 'rcept_dt']].copy()

df = df[df['account_nm'].isin(need_account)]

# thstrm_amount를 숫자형으로 변환
df['thstrm_amount'] = df['thstrm_amount'].astype(str)
df['thstrm_amount'] = (
    df['thstrm_amount']
    .str.replace(',', '', regex=False)
)
df['thstrm_amount'] = pd.to_numeric(df['thstrm_amount'], errors='coerce')

# frmtrm_amount를 숫자형으로 변환
df['frmtrm_amount'] = df['frmtrm_amount'].astype(str)
df['frmtrm_amount'] = (
    df['frmtrm_amount']
    .str.replace(',', '', regex=False)
)
df['frmtrm_amount'] = pd.to_numeric(df['frmtrm_amount'], errors='coerce')

df['thstrm_amount'].fillna(df['frmtrm_amount'], inplace=True)

reprt_code_to_num = {'11013': 1, '11012': 2, '11014': 3, '11011': 4}
df['quarter_num'] = df['reprt_code'].map(reprt_code_to_num)

df['temp_sort_key'] = df['bsns_year'].astype(int) * 4 + df['quarter_num']

df.sort_values(by=['corp_name', 'account_nm', 'temp_sort_key'], inplace=True)

# 다음 해 동일 분기 frmtrm_amount로 thstrm_amount 채우는 로직
account_of_interest = '당기순이익(손실)'
mask_for_interest = (df['account_nm'] == account_of_interest) & (df['thstrm_amount'].isna())

target_accounts_for_q4_solo = ['매출액', '영업이익', '당기순이익(손실)']
for account in target_accounts_for_q4_solo:
    df = calculate_q4_solo_amount(df, account)


if mask_for_interest.any():
    # next_quarter_year 변수를 재정의하여 다음 해(N+1)로 설정
    next_quarter_year = df['bsns_year'].astype(int).copy() + 1

    # next_quarter_num 변수를 현재 분기 번호(Qn)로 설정
    next_quarter_num = df['quarter_num'].copy()

    next_temp_sort_key = next_quarter_year * 4 + next_quarter_num

    df['lookup_id'] = df['corp_name'].astype(str) + '_' + df['account_nm'].astype(str) + '_' + df['temp_sort_key'].astype(str)

    frmtrm_amount_lookup_map = df.set_index('lookup_id')['frmtrm_amount'].to_dict()

    lookup_id_for_fill = df['corp_name'].astype(str) + '_' + df['account_nm'].astype(str) + '_' + next_temp_sort_key.astype(str)

    fill_values = lookup_id_for_fill.map(frmtrm_amount_lookup_map)

    df.loc[mask_for_interest, 'thstrm_amount'] = df.loc[mask_for_interest, 'thstrm_amount'].fillna(fill_values)

df.drop(columns=['quarter_num', 'temp_sort_key', 'lookup_id'], inplace=True, errors='ignore')

df['thstrm_amount'] = df.groupby(['corp_name', 'account_nm'])['thstrm_amount'].ffill()

df['thstrm_amount'].fillna(0, inplace=True)

df['frmtrm_amount'].fillna(0, inplace=True)

df['reprt_code'] = df['reprt_code'].astype(str).str.strip()

df['year_quarter'] = df['bsns_year'].astype(str) + '-' + df['reprt_code'].apply(year_quarter_classifier).fillna('')
df['year_quarter'] = df['year_quarter'].astype(str)

df.drop_duplicates(
    subset=['corp_name', 'year_quarter', 'account_nm'],
    inplace=True
)

df.set_index(df['corp_name'] + '-' + df['year_quarter'] + '-' + df['account_nm'], inplace=True)
df.index.name = 'corp_yqan'

df.sort_values(['corp_name','year_quarter'], inplace=True)

finance_filepath = '/content/AllCompanies_Finance.csv'
df.to_csv(finance_filepath, encoding='utf-8-sig')

print(df)
```

## pykrx 설치
```
!pip install pykrx
```
# 주가 데이터 확보
### 넷마블의 주식 코드 결측치를 채우고 pykrx를 통해 주가 데이터를 받아옵니다
```
from pykrx import stock
import pandas as pd

all_stock_data = []

corp_target_df.replace(' ', np.nan, inplace=True)
corp_target_df.dropna(subset=['stock_code'], inplace=True)
corp_codes = corp_target_df['stock_code'].unique()

print(corp_target_df)

for stock_code in corp_codes:
    # 회사명 가져오기
    corp_name = corp_target_df.loc[corp_target_df['stock_code'] == stock_code, 'corp_name'].iloc[0]
    print(stock_code, '-', corp_name)

    # 2018년부터 2025년 3분기까지의 주가 데이터 가져오기
    df = stock.get_market_ohlcv("20180101", "20250930", stock_code)

    # 결측치 제거
    df.dropna(how='any', inplace=True)

    # 회사명과 종목코드 추가
    df['corp_name'] = corp_name
    df['stock_code'] = stock_code
    df['날짜'] = df.index

    # 전체 통합용 리스트에 추가
    all_stock_data.append(df)

# 모든 회사 데이터를 하나로 합치기
final_stock_df = pd.concat(all_stock_data).reset_index(drop=True)

# 전체 통합 CSV 저장 (finance와 동일하게 하나의 파일만)
final_stock_filepath = '/content/AllCompanies_Stock.csv'
final_stock_df.to_csv(final_stock_filepath, encoding='utf-8-sig', index=False)

print(final_stock_df.head())
```

# 5. 주가-실적 상관관계 시각화
### 전처리된 재무 데이터와 실제 주가 흐름을 비교하는 그래프를 생성. 기업별 주가 지수화(Indexing)를 통해 실적 변동과 주가 추이의 괴리 및 동조화 현상을 시각적으로 분석합니다.
```
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 주가 데이터 전처리 및 주기별(일/월/분기) 시각화 함수
def draw_stock_graph(fig, stock_df, corp_name):
    stock = stock_df[stock_df['corp_name'] == corp_name].copy()
    stock['날짜'] = pd.to_datetime(stock['날짜'])
    stock.set_index('날짜', inplace=True)

    monthly_close = stock['종가'].resample('ME').last()
    quarter_close = stock['종가'].resample('QE').last()

    fig.add_trace(go.Scatter(
        x=stock.index, y=stock['종가'],
        mode='lines',
        name='일별 종가',
        legendgroup='stock',
        line=dict(color='lightblue')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=monthly_close.index, y=monthly_close.values,
        mode='lines+markers',
        name='월별 종가',
        legendgroup='stock',
        line=dict(color='darkblue'),
        marker=dict(symbol='circle', size=6)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=quarter_close.index, y=quarter_close.values,
        mode='markers',
        name='분기별 종가',
        legendgroup='stock',
        marker=dict(color='red', size=8, symbol='circle')
    ), row=1, col=1)


# 재무 데이터 전처리 및 주요 계정(영업이익/순이익) 시각화 함수
def draw_finance_graph(fig, finance_df, corp_name):
    # 데이터 필터링 및 분기 정렬
    finance = finance_df[finance_df['corp_name'] == corp_name].copy()

    finance['year_quarter'] = finance['year_quarter'].astype(str)

    finance = finance[finance['year_quarter'] != ''].copy()

    finance['year_quarter_sort_key'] = finance['year_quarter'].str.split('-').str[0].astype(int) * 4 + \
                                       finance['year_quarter'].str.split('-').str[1].map({'Q1':1, 'Q2':2, 'Q3':3, 'Q4':4})
    finance = finance.sort_values(by='year_quarter_sort_key').drop(columns='year_quarter_sort_key')

    # 주요 계정 추출 및 단위 변환 (억 원)
    매출액 = finance[finance['account_nm'] == '매출액'].copy()
    영업이익 = finance[finance['account_nm'] == '영업이익'].copy()
    당기순이익 = finance[finance['account_nm'] == '당기순이익(손실)'].copy()

    매출액['thstrm_amount'] = pd.to_numeric(매출액['thstrm_amount'], errors='coerce')
    영업이익['thstrm_amount'] = pd.to_numeric(영업이익['thstrm_amount'], errors='coerce')
    당기순이익['thstrm_amount'] = pd.to_numeric(당기순이익['thstrm_amount'], errors='coerce')

    매출액['thstrm_amount'] = 매출액['thstrm_amount'] / 100000000
    영업이익['thstrm_amount'] = 영업이익['thstrm_amount'] / 100000000
    당기순이익['thstrm_amount'] = 당기순이익['thstrm_amount'] / 100000000

    # 그래프 생성 (순이익 바 차트, 영업이익 라인 차트)
    colors = ['orangered' if x >= 0 else 'royalblue' for x in 당기순이익['thstrm_amount']]

    fig.add_trace(go.Bar(
        x=당기순이익['year_quarter'],
        y=당기순이익['thstrm_amount'],
        name='순이익/순손실',
        legendgroup='finance',
        marker_color=colors
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=영업이익['year_quarter'],
        y=영업이익['thstrm_amount'],
        mode='lines+markers',
        name='분기별 영업이익',
        legendgroup='finance',
        line=dict(color='limegreen'),
        marker=dict(symbol='circle', size=6)
    ), row=1, col=2)


if __name__ == '__main__':
    # 데이터 파일 로드
    stock_df = pd.read_csv('/content/AllCompanies_Stock.csv', parse_dates=['날짜'])
    finance_df = pd.read_csv('/content/AllCompanies_Finance.csv')

    # 기업별 그래프 생성 및 저장 루프
    for _, row in corp_target_df.iterrows():
        corp_name = row['corp_name']

        # 서브플롯 레이아웃 설정 (1행 2열)
        fig = make_subplots(
            rows=1, cols=2,
            shared_xaxes=False,
            subplot_titles=(corp_name + ' 일/월/분기별 주가',
                            corp_name + ' 분기별 영업이익')
        )

        # 각 그래프 그리기 함수 호출
        draw_stock_graph(fig, stock_df, corp_name)
        draw_finance_graph(fig, finance_df, corp_name)

        # 전체 레이아웃 및 스타일 설정
        fig.update_layout(
            width=1800,
            height=600,
            title=dict(
                text=corp_name + ' 주가 및 재무제표 그래프',
                x=0.5
            ),
            xaxis_title='년도',
            legend=dict(
                x=-0.1,
                y=0.5,
                xanchor='center',
                yanchor='middle',
                orientation='v',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='Black',
                borderwidth=1
            ),
            margin=dict(l=150)
        )

        # Y축 단위 및 포맷 설정
        fig.update_yaxes(
            title_text='주가 (원)',
            tickformat=',d',
            row=1, col=1
        )
        fig.update_yaxes(
            title_text='금액 (억 원)',
            tickformat=',d',
            row=1, col=2
        )
        
        # 결과 출력 및 HTML 파일 저장
        fig.show()
        fig.write_html(f'/content/{corp_name}plot.html')
```
## 주가 지표 분석용
```
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 데이터 로드 및 전처리
stock_df = pd.read_csv('/content/AllCompanies_Stock.csv', parse_dates=['날짜'])
finance_df = pd.read_csv('/content/AllCompanies_Finance.csv')

# 1. 재무 데이터 정렬 키 생성 (중요: 시간 순 정렬을 위함)
finance_df['year_quarter'] = finance_df['year_quarter'].astype(str)
finance_df = finance_df[finance_df['year_quarter'] != 'nan']

# 정렬용 키: 연도 * 4 + 분기 숫자 (예: 2020-Q1 -> 8081)
finance_df['sort_key'] = finance_df['year_quarter'].str.split('-').str[0].astype(int) * 4 + \
                         finance_df['year_quarter'].str.split('-').str[1].map({'Q1':1, 'Q2':2, 'Q3':3, 'Q4':4})

# 2. 영업이익 데이터 추출 및 단위 변환 (억 원)
profit_df = finance_df[finance_df['account_nm'] == '영업이익'].copy()
profit_df['thstrm_amount'] = pd.to_numeric(profit_df['thstrm_amount'], errors='coerce').fillna(0) / 100000000

# 회사별 고유 색상 매핑
corp_color_map = {
    '넥슨게임즈': '#1f77b4', '엔씨소프트': '#ff7f0e',
    '크래프톤': '#2ca02c', '카카오게임즈': '#d62728', '넷마블': '#9467bd'
}
fallback_colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def draw_period_comparison(period_name, start_date, end_date, start_key, end_key):
    # 기간 필터링
    period_stock = stock_df[(stock_df['날짜'] >= start_date) & (stock_df['날짜'] <= end_date)].sort_values('날짜')
    period_profit = profit_df[(profit_df['sort_key'] >= start_key) & (profit_df['sort_key'] <= end_key)]
    
    # 3. X축 순서 강제 고정 
    sorted_quarters = period_profit[['year_quarter', 'sort_key']].drop_duplicates().sort_values('sort_key')['year_quarter'].tolist()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{period_name} 주가 (분기간 지수화)', f'{period_name} 영업이익'),
        horizontal_spacing=0.15
    )

    # 그리기 순서: 리스트 역순 (넷마블 먼저 그려짐)
    corp_list = period_stock['corp_name'].unique()[::-1]

    for i, corp in enumerate(corp_list):
        corp_stock = period_stock[period_stock['corp_name'] == corp]
        corp_prof = period_profit[period_profit['corp_name'] == corp].sort_values('sort_key')
        
        line_color = corp_color_map.get(corp, fallback_colors[i % len(fallback_colors)])

        # [좌측] 주가 그래프: 월간 지수화 + 곡선(Spline)
        if not corp_stock.empty:
            corp_stock_monthly = corp_stock.set_index('날짜').resample('QE')['종가'].last().reset_index()
            if not corp_stock_monthly.empty:
                start_price = corp_stock_monthly['종가'].iloc[0]
                normalized_price = (corp_stock_monthly['종가'] / start_price) * 100

                fig.add_trace(go.Scatter(
                    x=corp_stock_monthly['날짜'], y=normalized_price,
                    mode='lines',
                    name=f'{corp} 주가',
                    legendgroup=corp,
                    line=dict(color=line_color, shape='spline', width=2), # 곡선 처리
                    hovertemplate='%{y:.1f}<extra></extra>'
                ), row=1, col=1)

        # [우측] 영업이익 그래프
        if not corp_prof.empty:
            fig.add_trace(go.Scatter(
                x=corp_prof['year_quarter'], y=corp_prof['thstrm_amount'],
                mode='lines+markers',
                name=f'{corp} 영업이익',
                legendgroup=corp,
                line=dict(color=line_color, width=2),
                marker=dict(size=8, color=line_color)
            ), row=1, col=2)

    # 레이아웃 설정
    fig.update_layout(
        title_text=f"{period_name} 기업 비교 분석",
        width=1800, height=700,
        hovermode="x unified"
    )
    
    # Y축 설정
    fig.update_yaxes(title_text="주가 지수 (Start=100)", row=1, col=1)
    fig.update_yaxes(title_text="영업이익 (억 원)", tickformat=',d', row=1, col=2)

    # [핵심] X축 정렬 강제 적용 (우측 그래프)
    # 카테고리 순서를 sorted_quarters 리스트 순서대로 강제함
    fig.update_xaxes(type='category', categoryorder='array', categoryarray=sorted_quarters, row=1, col=2)

    fig.show()
    fig.write_html(f'/content/Comparison_Profit_{period_name}_Fixed.html')

# 실행: 2020~2022
draw_period_comparison("2020_Q1~2022_Q2", '2020-01-01', '2022-06-30', 8081, 8090)

# 실행: 2022~2024
draw_period_comparison("2022~2024", '2022-01-01', '2024-12-31', 8089, 8100)
```
## 22년 이후의 주가 하락 원인을 분석하기 위한 소비자 지수 분석 (반드시 xslx파일을 다운받으셔야해요)
```
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 데이터 파일 로드 및 전처리
df_raw = pd.read_excel('/content/422601_20251205012846334_excel.xlsx', engine='openpyxl', skiprows=[0, 1], header=0) 

# 인덱스 설정 (첫 번째 컬럼을 항목명으로 지정)
df_raw = df_raw.set_index(df_raw.columns[0])
df_raw.index.name = 'Category'

# 필요한 항목명(지수)을 찾기 위한 함수
def find_category(keyword, category_list):
    # TypeError 방지를 위해 문자열인 항목만 필터링
    valid_categories = [c for c in category_list if isinstance(c, str)]
    
    matches = [c for c in valid_categories if keyword in c and '상승률' not in c and '%' not in c]
    if matches:
        # 가장 짧은 항목명을 선택 (주요 지수 항목일 가능성 높음)
        return min(matches, key=len)
    return None

# CPI 지수 항목명 추출
all_categories = df_raw.index.tolist()
food_col = find_category('식료품', all_categories)
clothing_col = find_category('의류', all_categories)
housing_col = find_category('주거', all_categories)
entertainment_col = find_category('오락', all_categories)
total_cpi_col = find_category('총지수', all_categories)

# CPI 지수 데이터만 필터링 및 전치(Transpose)
# 유효한 항목명만 사용하여 필터링
valid_cols = [food_col, clothing_col, housing_col, entertainment_col, total_cpi_col]
index_rows = [name for name in all_categories if name in valid_cols and name is not None]

# 항목명을 명시적으로 지정하여 loc 사용
df_cpi_index = df_raw.loc[index_rows].copy()

# 컬럼(연도) 이름 정리 및 전치
df_cpi_index.columns = df_cpi_index.columns.astype(str).str.strip()
df_plot = df_cpi_index.transpose()
df_plot.index.name = 'Year'

# 2018년부터 2024년까지 데이터 필터링
df_plot = df_plot.loc['2018':'2024'].copy()
df_plot.index = df_plot.index.astype(str)

# 데이터 컬럼을 숫자형으로 변환
for col in df_plot.columns:
    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
    
# 최종 플롯에 사용할 항목 정의
categories_map = {
    '식료품 및 비주류음료 (식)': food_col,
    '의류 및 신발 (의)': clothing_col,
    '주거·수도·전기 및 연료 (주)': housing_col,
    '오락 및 문화 (게임/오락)': entertainment_col,
}

# 필요한 컬럼만 선택
cols_to_use = list(categories_map.values()) + [total_cpi_col]
cols_to_use = list(set([c for c in cols_to_use if c is not None]))
df_final = df_plot[cols_to_use].copy()
    
# 그래프 생성 준비 (2개의 Y축)
title = "의식주, 오락 지수와 소비자물가 총지수 비교"
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 1. 라인 플롯 (의식주, 오락 지수) - 좌측 Y축
for label, col_name in categories_map.items():
    if col_name in df_final.columns and col_name != total_cpi_col:
        fig.add_trace(go.Scatter(
            x=df_final.index,
            y=df_final[col_name],
            mode='lines+markers',
            name=label,
            line=dict(width=3),
            marker=dict(size=8)
        ), secondary_y=False)

# 2. 바 플롯 (소비자물가 총지수) - 우측 Y축
if total_cpi_col in df_final.columns:
    fig.add_trace(go.Bar(
        x=df_final.index,
        y=df_final[total_cpi_col],
        name='소비자물가 총지수',
        marker_color='rgba(152, 0, 0, 0.4)',
        opacity=0.6,
    ), secondary_y=True)

# 레이아웃 및 축 설정
fig.update_layout(
    title_text=f'<b>{title}</b><br><sup>(기준년도 2020 = 100)</sup>',
    title_x=0.5,
    xaxis_title="연도",
    legend_title="지수 항목",
    template="plotly_white",
    height=650,
    width=1100
)

# 좌측 Y축 (라인 플롯) 설정
fig.update_yaxes(
    title_text="<b>품목별 소비자물가 지수</b> (2020=100)",
    secondary_y=False,
    tickformat=".1f",
    gridcolor='rgba(0,0,0,0.1)'
)

# 우측 Y축 (바 플롯) 설정
fig.update_yaxes(
    title_text="<b>소비자물가 총지수</b> (2020=100)",
    secondary_y=True,
    showgrid=False,
    tickformat=".1f",
    # 총지수 바가 너무 작아지지 않도록 축 범위 자동 조정
    range=[df_final[total_cpi_col].min() * 0.9, df_final[total_cpi_col].max() * 1.05]
)

# 바 플롯이 라인 플롯 뒤에 그려지도록 순서 조정
fig.data = fig.data[-1:] + fig.data[:-1]

fig.show()
```

## 혹시 모를 matplot 한글 오류를 위해 한글 폰트를 설치합니다.
```
# 단계 1: 폰트 설치
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

!apt-get -qq -y install fonts-nanum > /dev/null
#fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'



fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumGothic')                        # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에 폰트 추가
plt.rcParams.update({'font.size': 18, 'font.family': 'NanumGothic'}) # 폰트 설
```
```
# 단계 2: 런타임 재시작
import os
os.kill(os.getpid(), 9)
```
```
# 단계 3: 한글 폰트 설정
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

# 마이너스 표시 문제
mpl.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
    name='NanumGothic')                     
fm.fontManager.ttflist.insert(0, fe)         
plt.rcParams.update({'font.size': 18, 'font.family': 'NanumGothic'})
```

# 6. AI 하이브리드 예측 모델링 (LSTM + XGBoost)
## 1단계(LSTM): 과거 시계열 패턴을 학습해 미래(2025년 4분기)의 재무 실적을 선행 예측합니다.
## 2단계(XGBoost): 예측된 재무 지표와 기술적 주가 지표를 결합하여, 향후 단기 주가 흐름을 예측하는 최종 머신러닝 모델 구축합니다.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import math
import warnings
from pykrx.stock import get_market_ohlcv_by_date
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os
import sys

# 경고 메시지 출력 제한
warnings.filterwarnings('ignore')

# 설정 및 상수 정의
FINANCE_FILE = 'AllCompanies_Finance.csv'
RESULT_FILE_XGBOOST = 'xgboost_past_predictions_with_lstm_finance_10_11.csv'
PREDICT_DAYS = 5
LOOK_BACK = 15
STOCK_PULL_END_DATE = '2025-11-30'

# 학습 및 검증 기간 설정
TRAIN_START_CANDIDATES = ['2018-01-01', '2020-01-01', '2022-01-01']
VALIDATION_START_DATE = '2025-07-01'
FINAL_TRAIN_END_DATE = '2025-09-30'

# 테스트(예측 목표) 기간 설정
TEST_START_DATE = '2025-10-01'
TEST_END_DATE = '2025-11-29'

# 대상 기업 및 종목코드 정의
corp_target_data = [
    {'corp_name': '크래프톤', 'stock_code': '259960'},
    {'corp_name': '넥슨게임즈', 'stock_code': '225570'},
    {'corp_name': '카카오게임즈', 'stock_code': '293490'},
    {'corp_name': '엔씨소프트', 'stock_code': '036570'},
    {'corp_name': '넷마블', 'stock_code': '251270'}
]

# 한글 폰트 설치 및 설정 (Colab 환경)
try:
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 데이터 로드 및 전처리
def load_and_preprocess_data():
    try:
        finance_df = pd.read_csv(FINANCE_FILE)
        finance_df = finance_df[['corp_name', 'year_quarter', 'account_nm', 'thstrm_amount']]
        finance_df.rename(columns={'thstrm_amount': 'amount'}, inplace=True)

        def fetch_and_prepare_latest_data(corp_item):
            code = corp_item['stock_code']
            name = corp_item['corp_name']
            try:
                df = get_market_ohlcv_by_date(fromdate='2018-01-01', todate=STOCK_PULL_END_DATE, ticker=code).reset_index()
                df.columns = ['날짜', '시가', '고가', '저가', '종가', '거래량', '등락률']
                df['corp_name'] = name
                df['등락률'] = df['종가'].pct_change() * 100
                df.dropna(inplace=True)
                return df
            except Exception:
                return pd.DataFrame()

        all_stock_data = [fetch_and_prepare_latest_data(item) for item in corp_target_data]
        stock_df = pd.concat(all_stock_data, ignore_index=True)
        stock_df['날짜'] = pd.to_datetime(stock_df['날짜'])
        print("주가 및 재무 데이터 로드 완료")
        return stock_df, finance_df
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        sys.exit(1)

# 재무 지표 및 기술적 지표 생성 및 병합
def create_finance_features(stock_data, finance_data):
    finance_pivot = finance_data.pivot_table(
        index=['corp_name', 'year_quarter'], columns='account_nm', values='amount'
    ).reset_index()

    def get_quarter_end_date(yq):
        year, quarter = yq.split('-')
        if quarter == 'Q1': return pd.to_datetime(f'{year}-03-31')
        if quarter == 'Q2': return pd.to_datetime(f'{year}-06-30')
        if quarter == 'Q3': return pd.to_datetime(f'{year}-09-30')
        if quarter == 'Q4': return pd.to_datetime(f'{year}-12-31')
        return None

    finance_pivot['재무_기준일'] = finance_pivot['year_quarter'].apply(get_quarter_end_date)
    finance_pivot.dropna(subset=['재무_기준일'], inplace=True)

    merged_data = stock_data.copy()
    merged_data['날짜_temp'] = merged_data['날짜']
    merged_data = pd.merge_asof(
        merged_data.sort_values('날짜_temp'),
        finance_pivot[['corp_name', '재무_기준일', '매출액', '영업이익', '당기순이익(손실)', '부채총계']].sort_values('재무_기준일'),
        by='corp_name',
        left_on='날짜_temp',
        right_on='재무_기준일',
        direction='backward'
    )
    merged_data.drop(columns=['날짜_temp', '재무_기준일'], inplace=True)

    # 이익률 계산
    merged_data['영업이익률'] = np.where(merged_data['매출액'] != 0, merged_data['영업이익'] / merged_data['매출액'], 0)
    merged_data['순이익률'] = np.where(merged_data['매출액'] != 0, merged_data['당기순이익(손실)'] / merged_data['매출액'], 0)
    merged_data.dropna(subset=['영업이익률'], inplace=True)

    # 이동평균 및 변동성 계산
    merged_data['MA_5'] = merged_data.groupby('corp_name')['종가'].transform(lambda x: x.rolling(window=5).mean())
    merged_data['MA_20'] = merged_data.groupby('corp_name')['종가'].transform(lambda x: x.rolling(window=20).mean())
    merged_data['Volatility'] = merged_data.groupby('corp_name')['등락률'].transform(lambda x: x.rolling(window=LOOK_BACK).std())

    merged_data.dropna(inplace=True)
    merged_data.sort_values(by=['corp_name', '날짜'], inplace=True)

    return merged_data, finance_pivot[['corp_name', '재무_기준일', '매출액', '영업이익', '당기순이익(손실)', '부채총계']].copy()

# LSTM 기반 4분기 재무 제표 예측
def predict_q4_finance_lstm(finance_data_pivot, corp_name):
    target_cols = ['매출액', '영업이익', '당기순이익(손실)']
    predicted_q4_finance = {}

    for target_col in target_cols:
        corp_finance = finance_data_pivot[finance_data_pivot['corp_name'] == corp_name].copy()
        ts_data = corp_finance.set_index('재무_기준일')[target_col].dropna().sort_index()
        
        last_actual_value = ts_data.iloc[-1] if len(ts_data) > 0 else 0
        if len(ts_data) < 8:
            predicted_q4_finance[target_col] = last_actual_value
            continue
            
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(ts_data.values.reshape(-1, 1))

        TIME_STEP = 4 
        X, y = [], []
        for i in range(len(scaled_data) - TIME_STEP):
            X.append(scaled_data[i:(i + TIME_STEP), 0])
            y.append(scaled_data[i + TIME_STEP, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(TIME_STEP, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        callback = EarlyStopping(monitor='loss', patience=10, verbose=0)
        
        try:
            model.fit(X, y, epochs=100, batch_size=1, verbose=0, callbacks=[callback])
            
            last_4_quarters = scaled_data[-TIME_STEP:].reshape(1, TIME_STEP, 1)
            predicted_scaled = model.predict(last_4_quarters, verbose=0)
            predicted_value = scaler.inverse_transform(predicted_scaled)[0, 0]
            
            # 예측값 유효성 검증 및 예외 처리
            if target_col == '매출액':
                if predicted_value < 1_000_000_000: 
                    predicted_q4_finance[target_col] = last_actual_value
                else:
                    predicted_q4_finance[target_col] = predicted_value
            else:
                if predicted_value < -10_000_000_000_000: 
                     predicted_q4_finance[target_col] = last_actual_value
                else:
                     predicted_q4_finance[target_col] = predicted_value

        except Exception:
            predicted_q4_finance[target_col] = last_actual_value
            continue

    q3_debt = finance_data_pivot[
        (finance_data_pivot['corp_name'] == corp_name) & 
        (finance_data_pivot['재무_기준일'] == '2025-09-30')
    ]['부채총계'].iloc[0] if '2025-09-30' in finance_data_pivot['재무_기준일'].astype(str).values else 0
    
    predicted_q4_finance['부채총계'] = q3_debt
    
    formatted_finance = {k: f"{int(v):,}" for k, v in predicted_q4_finance.items()}
    
    return predicted_q4_finance, formatted_finance

# XGBoost 모델 학습 및 주가 예측 수행
def train_and_predict_xgb_with_lstm_finance(data, corp_name, predicted_finance):
    corp_data = data[data['corp_name'] == corp_name].copy()

    test_data_range = (corp_data['날짜'] >= TEST_START_DATE) & (corp_data['날짜'] <= TEST_END_DATE)

    # 예측된 재무 데이터 적용
    corp_data.loc[test_data_range, '매출액'] = predicted_finance.get('매출액', corp_data['매출액'].mean())
    corp_data.loc[test_data_range, '영업이익'] = predicted_finance.get('영업이익', corp_data['영업이익'].mean())
    corp_data.loc[test_data_range, '당기순이익(손실)'] = predicted_finance.get('당기순이익(손실)', corp_data['당기순이익(손실)'].mean())
    corp_data.loc[test_data_range, '부채총계'] = predicted_finance.get('부채총계', corp_data['부채총계'].mean())

    corp_data.loc[test_data_range, '영업이익률'] = np.where(corp_data.loc[test_data_range, '매출액'] != 0, corp_data.loc[test_data_range, '영업이익'] / corp_data.loc[test_data_range, '매출액'], 0)
    corp_data.loc[test_data_range, '순이익률'] = np.where(corp_data.loc[test_data_range, '매출액'] != 0, corp_data.loc[test_data_range, '당기순이익(손실)'] / corp_data.loc[test_data_range, '매출액'], 0)

    corp_data['Target_Close'] = corp_data['종가'].shift(-PREDICT_DAYS)
    corp_data.dropna(subset=['Target_Close'], inplace=True)

    features = ['종가', '거래량', '등락률', '영업이익률', '순이익률', '부채총계', 'MA_5', 'MA_20', 'Volatility']
    best_rmse_val = float('inf')
    best_train_start = None
    best_params_final = None

    # 최적 학습 기간 탐색
    for start_date in TRAIN_START_CANDIDATES:
        train_set = corp_data[(corp_data['날짜'] >= start_date) & (corp_data['날짜'] < VALIDATION_START_DATE)]
        validation_set = corp_data[(corp_data['날짜'] >= VALIDATION_START_DATE) & (corp_data['날짜'] <= FINAL_TRAIN_END_DATE)].copy()

        if len(train_set) < 50 or len(validation_set) == 0: continue

        X_train, y_train = train_set[features].values, train_set['Target_Close'].values
        X_val, y_val = validation_set[features].values, validation_set['Target_Close'].values

        scaler_X = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        scaler_y = MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        param_grid = {'n_estimators': [100, 300, 500], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]}
        tscv = TimeSeriesSplit(n_splits=3)

        xgb_model = XGBRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1, verbose=0)
        grid_search.fit(X_train_scaled, y_train_scaled)

        X_val_scaled = scaler_X.transform(X_val)
        val_predictions_inv = scaler_y.inverse_transform(grid_search.best_estimator_.predict(X_val_scaled).reshape(-1, 1)).flatten()
        current_rmse = math.sqrt(mean_squared_error(y_val, val_predictions_inv))

        if current_rmse < best_rmse_val:
            best_rmse_val = current_rmse
            best_train_start = start_date
            best_params_final = grid_search.best_params_

    if best_train_start is None: return None, None, None

    # 최종 모델 학습 및 테스트
    final_train_data = corp_data[(corp_data['날짜'] >= best_train_start) & (corp_data['날짜'] <= FINAL_TRAIN_END_DATE)]
    test_data = corp_data[(corp_data['날짜'] >= TEST_START_DATE) & (corp_data['날짜'] <= TEST_END_DATE)].copy()

    if len(test_data) == 0: return None, None, None

    X_train_final, y_train_final = final_train_data[features].values, final_train_data['Target_Close'].values
    X_test_full, y_test_full = test_data[features].values, test_data['Target_Close'].values

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_final)
    X_test_scaled = scaler_X.transform(X_test_full)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_final.reshape(-1, 1)).flatten()

    final_model = XGBRegressor(**best_params_final, random_state=42)
    final_model.fit(X_train_scaled, y_train_scaled)

    test_predictions_scaled = final_model.predict(X_test_scaled)
    test_predictions_inv = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()

    test_rmse = math.sqrt(mean_squared_error(y_test_full, test_predictions_inv))

    # 결과 데이터프레임 생성
    plot_df = pd.DataFrame({
        '예측_시점_날짜': test_data['날짜'].values,
        '실제_5일후_종가': y_test_full,
        '예측_5일후_종가': test_predictions_inv,
        'corp_name': corp_name,
        'RMSE': test_rmse,
        '최적_시작일': best_train_start,
        'Q4_재무_예측값': str({k: int(v) for k,v in predicted_finance.items()})
    })
    plot_df['결과_날짜'] = plot_df['예측_시점_날짜'] + pd.Timedelta(days=PREDICT_DAYS)

    return plot_df, best_params_final, test_rmse
```
```
def run_prediction_and_save():
    # 데이터 로드 및 전처리
    stock_df, finance_df = load_and_preprocess_data()
    merged_stock_data, finance_pivot_for_lstm = create_finance_features(stock_df, finance_df)
    final_plot_results = []

    print("모델 학습 및 예측 프로세스 시작")

    # 기업별 학습 및 예측 수행
    for corp_data_item in corp_target_data:
        corp_name = corp_data_item['corp_name']

        # LSTM 기반 4분기 재무 지표 예측
        predicted_q4_finance, formatted_finance = predict_q4_finance_lstm(finance_pivot_for_lstm, corp_name)

        # XGBoost 주가 예측 모델 학습 및 실행
        result_df, best_params, test_rmse = train_and_predict_xgb_with_lstm_finance(
            merged_stock_data, corp_name, predicted_q4_finance
        )

        # 결과 리스트 추가 및 진행 상황 출력
        if result_df is not None:
            final_plot_results.append(result_df)
            print(f"[{corp_name}] 예측 완료 - RMSE: {test_rmse:,.0f}")

    # 최종 결과 병합 및 CSV 파일 저장
    if final_plot_results:
        final_result_df = pd.concat(final_plot_results, ignore_index=True)
        final_result_df.to_csv(RESULT_FILE_XGBOOST, index=False)
        print(f"모든 예측 결과 저장 완료: {RESULT_FILE_XGBOOST}")
        return True
    return False

# 파일 존재 여부 확인 및 예측 실행
if os.path.exists(RESULT_FILE_XGBOOST):
    print(f"기존 파일 {RESULT_FILE_XGBOOST} 덮어쓰기 진행")

run_prediction_and_save()
```
### 한글 폰트 에러가 또 나는 경우 이 코드를 실행해주세요 ㅎㅎ
```
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import os

# Colab 환경 한글 폰트 설정 함수
def setup_colab_font():
    try:
        # 나눔 폰트 패키지 설치
        !apt-get -qq -y install fonts-nanum > /dev/null

        # 나눔바른고딕 폰트 경로 지정
        font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'

        # 폰트 매니저 등록 및 설정 적용
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            
            # Matplotlib 폰트 지정 및 마이너스 기호 깨짐 방지
            plt.rc('font', family=font_name)
            plt.rcParams['axes.unicode_minus'] = False
            print("한글 폰트 설정 완료")
        else:
            print("폰트 파일을 찾을 수 없음")
            
    except Exception as e:
        print(f"폰트 설정 오류: {e}")

# 설정 함수 실행
setup_colab_font()
```
# 머신러닝 결과 시각화
```
# 시각화 및 결과 출력 함수
def visualize_results():
    # 시각화 대상 파일 존재 여부 확인
    if not os.path.exists(RESULT_FILE_XGBOOST):
        print(f"오류: 파일({RESULT_FILE_XGBOOST})이 존재하지 않음")
        return

    # 결과 데이터 로드 및 날짜 형변환
    result_df = pd.read_csv(RESULT_FILE_XGBOOST)
    result_df['결과_날짜'] = pd.to_datetime(result_df['결과_날짜'])

    print("예측 결과 시각화 시작")

    # 기업별 예측 결과 추출 및 시각화 루프
    for corp_name in result_df['corp_name'].unique():
        corp_data = result_df[result_df['corp_name'] == corp_name].reset_index(drop=True)

        # 주요 예측 지표 및 재무 데이터 추출
        test_rmse = corp_data['RMSE'].iloc[0]
        best_start = corp_data['최적_시작일'].iloc[0]
        q4_finance_str = corp_data['Q4_재무_예측값'].iloc[0]

        q4_finance_dict = eval(q4_finance_str)
        formatted_finance = {k: f"{v:,}" for k, v in q4_finance_dict.items()}
        best_params_str = str(corp_data.iloc[0].filter(like='param').to_dict())

        # 예측 결과 텍스트 출력
        print("-" * 50)
        print(f"[{corp_name}] 10~11월 주가 예측 결과")
        print(f" Q4 재무 예측: 매출액 {formatted_finance.get('매출액', 'N/A')}, 영업이익 {formatted_finance.get('영업이익', 'N/A')}")
        print(f"               순이익 {formatted_finance.get('당기순이익(손실)', 'N/A')}, 부채 {formatted_finance.get('부채총계', 'N/A')}")
        print(f" 최적 학습 시작일: {best_start}")
        print(f" RMSE: {test_rmse:,.0f}원")
        print("-" * 50)

        # 주가 예측 비교 그래프 생성
        plt.figure(figsize=(14, 7))

        plt.plot(corp_data['결과_날짜'], corp_data['실제_5일후_종가'], label='실제 5일 후 종가', color='blue', linewidth=2)
        plt.plot(corp_data['결과_날짜'], corp_data['예측_5일후_종가'], label='XGBoost 예측 5일 후 종가', color='red', linestyle='--', linewidth=2)

        # X축 날짜 포맷 설정
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WEDNESDAY, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_minor_locator(mdates.DayLocator())

        # 그래프 스타일 및 레이블 설정
        plt.title(f'[{corp_name}] XGBoost 5일 후 주가 예측 비교 (2025년 10월~11월)')
        plt.xlabel('결과 날짜')
        plt.ylabel('종가 (원)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 그래프 이미지 저장 및 출력
        image_filename = f'XGBoost_Prediction_{corp_name}_LSTM_Finance_10_11.png'
        plt.savefig(image_filename)
        plt.show()

# 시각화 함수 실행
visualize_results()
```
