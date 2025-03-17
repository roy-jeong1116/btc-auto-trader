from utils import *

download_btc_price_data()
calculate_technical_indicators()
filter_data_by_date()

# 랜덤 시드 고정
set_random_seeds()

# 데이터 전처리
df, X_train, X_test, y_train, y_test, max_price, min_price = preprocess_data()

# KFold 교차 검증
avg_mse = kfold_cross_validation(X_train, y_train)
print(f"Average MSE across folds: {avg_mse}")

# 최종 모델 훈련
final_model = create_and_train_lstm(X_train, y_train)

# 테스트 성능 평가
test_mse, y_pred_test = evaluate_model(final_model, X_test, y_test)
print(f"Test MSE: {test_mse}")

# 다음 날 예측
next_day_close = predict_next_day_close(final_model, df, max_price, min_price)
print(f"Predicted CLOSE for the next day: {next_day_close}")