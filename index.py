import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 가상의 데이터 생성
# X는 라만 스펙트럼 데이터, y는 해당 물질의 비율
X = np.random.rand(100, 100)  # 100개의 스펙트럼과 각각 100개의 데이터 포인트로 가정
y = np.random.rand(100, 4)   # 4개의 물질에 대한 비율로 가정

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
