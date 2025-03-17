from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import talib as ta
import numpy as np
import tensorflow as tf
import random

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

# 기술적 지표
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

# 재현성을 위해 난수 고정
def set_random_seeds(seed=42):
    """
    난수 고정을 통해 결과의 재현성을 보장합니다.
    
    Parameters:
        seed (int): 난수 초기화 시 사용할 시드 값 (기본값: 42)
        
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 데이터 전처리 함수
def preprocess_data(filename='technical_data.csv'):
    """
    주어진 CSV 파일에서 데이터를 불러오고 전처리한 후, 훈련 데이터와 테스트 데이터를 반환합니다.
    
    Parameters:
        filepath (str): 데이터가 저장된 CSV 파일 경로 (기본값: 'technical_data.csv')
        
    Returns:
        df (DataFrame): 전처리된 전체 데이터프레임
        X_train (ndarray): 훈련용 입력 데이터
        X_test (ndarray): 테스트용 입력 데이터
        y_train (ndarray): 훈련용 목표값 (CLOSE)
        y_test (ndarray): 테스트용 목표값 (CLOSE)
        max_price (float): 최대 CLOSE 값
        min_price (float): 최소 CLOSE 값
    """
    # 데이터 불러오기
    df = pd.read_csv(filename, parse_dates=['DATE'], index_col='DATE')

    # max, min 구하기
    max_price = df['CLOSE'].max()
    min_price = df['CLOSE'].min()

    # 스케일링
    scaler = MinMaxScaler()
    df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'SMA_14', 'RSI_14', 'BOLLINGER_UPPER', 'BOLLINGER_MIDDLE', 'BOLLINGER_LOWER',
        'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ATR', 'STDDEV', 'PIVOT', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3', 'ENGULFING', 'DOJI',
        'HAMMER', 'MORNINGSTAR', 'OBV']] = scaler.fit_transform(df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'SMA_14', 'RSI_14', 
                                                                   'BOLLINGER_UPPER', 'BOLLINGER_MIDDLE', 'BOLLINGER_LOWER', 'MACD',
                                                                   'MACD_SIGNAL', 'MACD_HIST', 'ATR', 'STDDEV', 'PIVOT', 'R1', 'S1', 
                                                                   'R2', 'S2', 'R3', 'S3', 'ENGULFING', 'DOJI', 'HAMMER', 'MORNINGSTAR', 'OBV']])
    
    # 상관관계가 높은 피처 제거
    corr_matrix = df.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.8) and column != 'CLOSE']
    df = df.drop(to_drop, axis=1)

    # 훈련 및 테스트 데이터 분리
    X = df.drop('CLOSE', axis=1)
    y = df['CLOSE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 데이터 reshape
    X_train = X_train.values.reshape(-1, 1, X_train.shape[1])
    X_test = X_test.values.reshape(-1, 1, X_test.shape[1])

    return df, X_train, X_test, y_train, y_test, max_price, min_price

# LSTM 모델 생성 및 훈련 함수
def create_and_train_lstm(X_train, y_train, epochs=100, batch_size=32):
    """
    LSTM 모델을 생성하고 훈련합니다.
    
    Parameters:
        X_train (ndarray): 훈련용 입력 데이터
        y_train (ndarray): 훈련용 목표값 (CLOSE)
        epochs (int): 훈련의 epoch 수 (기본값: 100)
        batch_size (int): 배치 사이즈 (기본값: 32)
        
    Returns:
        model (keras.Sequential): 훈련된 LSTM 모델
    """
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    return model

# KFold 교차검증 및 MSE 계산 함수
def kfold_cross_validation(X_train, y_train, k=5, epochs=100, batch_size=32):
    """
    KFold 교차 검증을 통해 모델을 평가하고 평균 MSE를 계산합니다.
    
    Parameters:
        X_train (ndarray): 훈련용 입력 데이터
        y_train (ndarray): 훈련용 목표값 (CLOSE)
        k (int): KFold의 분할 수 (기본값: 5)
        epochs (int): 훈련의 epoch 수 (기본값: 100)
        batch_size (int): 배치 사이즈 (기본값: 32)
        
    Returns:
        mean_mse (float): KFold 교차 검증을 통한 평균 MSE
    """
    kf = KFold(n_splits=k, shuffle=True)
    mse_values = []

    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train.values[train_index], y_train.values[test_index]

        model = create_and_train_lstm(X_train_fold, y_train_fold, epochs, batch_size)
        
        y_pred_fold = model.predict(X_test_fold).flatten()
        mse = mean_squared_error(y_test_fold, y_pred_fold)
        mse_values.append(mse)

    return np.mean(mse_values)

# 모델 예측 및 성능 평가 함수
def evaluate_model(model, X_test, y_test):
    """
    훈련된 모델을 사용하여 테스트 데이터에서 예측을 수행하고, 성능 평가(MSE)를 반환합니다.
    
    Parameters:
        model (keras.Sequential): 훈련된 LSTM 모델
        X_test (ndarray): 테스트용 입력 데이터
        y_test (ndarray): 테스트용 목표값 (CLOSE)
        
    Returns:
        test_mse (float): 테스트 데이터에서의 MSE
        y_pred_test (ndarray): 테스트 데이터에 대한 예측값
    """
    y_pred_test = model.predict(X_test).flatten()
    test_mse = mean_squared_error(y_test.values, y_pred_test)
    return test_mse, y_pred_test

# 마지막 날의 데이터를 기반으로 다음 날 CLOSE 예측 함수
def predict_next_day_close(model, df, max_price, min_price):
    """
    마지막 날의 데이터를 기반으로 다음 날의 CLOSE 값을 예측합니다.
    
    Parameters:
        model (keras.Sequential): 훈련된 LSTM 모델
        df (DataFrame): 전처리된 데이터프레임
        max_price (float): 최대 CLOSE 값
        min_price (float): 최소 CLOSE 값
        
    Returns:
        next_day_close (float): 예측된 다음 날의 CLOSE 값 (역변환된 값)
    """
    last_day_data = df.iloc[-1].drop('CLOSE').values.reshape(1, 1, -1)  # 마지막 날 데이터 준비
    next_day_close_scaled = model.predict(last_day_data).flatten()[0]  # 스케일된 값 예측
    next_day_close = next_day_close_scaled * (max_price - min_price) + min_price
    return next_day_close