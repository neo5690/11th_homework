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


####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

