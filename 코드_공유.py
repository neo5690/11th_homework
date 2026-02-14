# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
# 데이터 불러오기
wine = load_wine()

# DataFrame으로 변환
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# target 컬럼 추가
df['target'] = wine.target

# feature 데이터 (target 컬럼 제거)
X = df.drop('target', axis=1)

# target 데이터
y = df['target']

# train / test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 확인
print("X_train 크기:", X_train.shape)
print("X_test 크기:", X_test.shape)
print("y_train 크기:", y_train.shape)
print("y_test 크기:", y_test.shape)


####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 모델 생성
dt = DecisionTreeClassifier(random_state=42)

# 하이퍼파라미터 후보 수정
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearch 설정
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 학습
grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 출력
print("Best parameters:", grid_search.best_params_)

# 테스트 데이터 평가
best_model1 = grid_search.best_estimator_
y_pred = best_model1.predict(X_test)

print("Best accuracy:", accuracy_score(y_test, y_pred))

# Feature Importance 시각화
importances = best_model1.feature_importances_
feature_names = X_train.columns

indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])

plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")

plt.tight_layout()
plt.show()



####### B 작업자 작업 수행 ########

''' 코드 작성 바랍니다 '''
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 모델 생성
xgb_model = XGBClassifier(
    objective='multi:softprob',  # 다중분류
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# 하이퍼파라미터 후보 설정
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

# GridSearch 설정
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# 모델 학습
grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 출력
print("Best parameters:", grid_search.best_params_)

# 테스트 데이터 평가
best_model2 = grid_search.best_estimator_
y_pred = best_model2.predict(X_test)

print("Best accuracy:", accuracy_score(y_test, y_pred))



# XGBClassifier 모델에서 Feature Importance 가져오기
importances = best_model2.feature_importances_
feature_names = X_train.columns

# 중요도 내림차순 정렬
indices = importances.argsort()[::-1]

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])

plt.xticks(range(len(importances)), feature_names[indices], rotation=45)  
plt.xlabel("Feature")   
plt.ylabel("Importance") 
plt.title("Feature Importance")

plt.tight_layout()
plt.show()

