import yfinance as yf
import pandas as pd
import talib as ta
from datetime import datetime, timedelta

# OHLCV
def download_btc_price_data(period='4y', interval='1d', filename='technical_data.csv'):
    """
    비트코인 가격 데이터를 다운로드하여 저장하는 함수
    
    Parameters:
        period (str): 데이터 기간 (예: '4y' -> 4년치 데이터)
        interval (str): 데이터 간격 (예: '1d' -> 일간 데이터)
        filename (str): 저장할 CSV 파일 이름 (기본값: 'technical_data.csv')

    Returns:
        None
    """
    # 비트코인 가격 데이터 다운로드
    btc_ohlcv = yf.download('BTC-USD', period=period, interval=interval)

    # 컬럼명 설정 및 재정렬
    btc_ohlcv.columns = ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']
    btc_ohlcv = btc_ohlcv[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]

    # 인덱스 이름을 'DATE'로 변경
    btc_ohlcv.index.name = 'DATE'

    # CSV 파일로 저장
    btc_ohlcv.to_csv(filename, index=True)

import pandas as pd
import talib as ta
from datetime import datetime, timedelta

def calculate_technical_indicators(filename='technical_data.csv'):
    """
    기술적 지표를 계산하고, 결과를 업데이트하는 함수.
    
    - SMA: 단순 이동평균 (14일 기준)
    - RSI: 상대 강도 지수 (14일 기준)
    - Bollinger Bands: 볼린저 밴드 (20일 기준)
    - MACD: 이동 평균 수렴 발산 (fast=12, slow=26, signal=9)
    - ATR: 평균 진폭 (14일 기준)
    - STDDEV: 표준편차 (14일 기준)
    - Support/Resistance: 피벗 포인트 기반의 지지선과 저항선 계산
    - Candlestick Patterns: 주요 캔들 패턴 (Engulfing, Doji, Hammer, MorningStar)
    - OBV: 잉여 거래량
    
    Parameters:
        filename (str): 데이터가 저장된 CSV 파일 이름 (기본값: 'technical_data.csv')
    
    Returns:
        None
    """
    # CSV 파일 읽기
    df = pd.read_csv(filename, index_col='DATE', parse_dates=True)

    # SMA (단순 이동평균, 14일 기준)
    df['SMA_14'] = ta.SMA(df['CLOSE'], timeperiod=14)
    
    # RSI (상대 강도 지수, 14일 기준)
    df['RSI_14'] = ta.RSI(df['CLOSE'], timeperiod=14)
    
    # Bollinger Bands (볼린저 밴드, 20일 기준, 상한선: 2배 표준편차, 하한선: 2배 표준편차)
    upper_band, middle_band, lower_band = ta.BBANDS(df['CLOSE'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BOLLINGER_UPPER'] = upper_band
    df['BOLLINGER_MIDDLE'] = middle_band
    df['BOLLINGER_LOWER'] = lower_band
    
    # MACD (이동 평균 수렴 발산, fast=12, slow=26, signal=9)
    macd, macd_signal, macd_hist = ta.MACD(df['CLOSE'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_SIGNAL'] = macd_signal
    df['MACD_HIST'] = macd_hist
    
    # ATR (평균 진폭, 14일 기준)
    df['ATR'] = ta.ATR(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=14)
    
    # STDDEV (표준편차, 14일 기준)
    df['STDDEV'] = ta.STDDEV(df['CLOSE'], timeperiod=14)
    
    # Support & Resistance (피벗 포인트 기반, R1, S1, R2, S2, R3, S3 계산)
    df['PIVOT'] = (df['HIGH'] + df['LOW'] + df['CLOSE']) / 3
    df['R1'] = (2 * df['PIVOT']) - df['LOW']
    df['S1'] = (2 * df['PIVOT']) - df['HIGH']
    df['R2'] = df['PIVOT'] + (df['HIGH'] - df['LOW'])
    df['S2'] = df['PIVOT'] - (df['HIGH'] - df['LOW'])
    df['R3'] = df['HIGH'] + 2 * (df['PIVOT'] - df['LOW'])
    df['S3'] = df['LOW'] - 2 * (df['HIGH'] - df['PIVOT'])
    
    # Candlestick Patterns (주요 캔들 패턴: Engulfing, Doji, Hammer, MorningStar)
    df['ENGULFING'] = ta.CDLENGULFING(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE']) // 100
    df['DOJI'] = ta.CDLDOJI(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE']) // 100
    df['HAMMER'] = ta.CDLHAMMER(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE']) // 100
    df['MORNINGSTAR'] = ta.CDLMORNINGSTAR(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE']) // 100
    
    # OBV (잉여 거래량)
    df['OBV'] = ta.OBV(df['CLOSE'], df['VOLUME'])
    
    # CSV 파일 업데이트
    df.to_csv(filename, mode='w', header=True)


# 날짜 필터링
def filter_data_by_date(filename='technical_data.csv'):
    """
    주어진 DataFrame에서 현재 날짜 기준으로 지정된 년수만큼 뒤의 데이터만 필터링하는 함수.
    
    Parameters:
        filename (str): 데이터가 저장된 CSV 파일 이름 (기본값: 'technical_data.csv')    
    
    Returns:
        None
    """
    # CSV 파일 읽기
    df = pd.read_csv(filename, index_col='DATE', parse_dates=True)

    # 현재 날짜 기준으로 지정된 년수만큼 뒤의 날짜 계산
    three_years_ago = (datetime.now() - timedelta(days=1)).replace(year=datetime.now().year - 3)
    
    # 날짜 인덱스가 3년 전 이후인 데이터만 필터링
    df = df[df.index >= three_years_ago]

    # CSV 파일 업데이트
    df.to_csv(filename, mode='w', header=True)