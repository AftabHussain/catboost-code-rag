import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor

df = pd.read_csv('zillow_sample_11.csv')
scaler = MinMaxScaler()
scaled_cols = ['sqft_living', 'sqft_lot']
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

X = df[scaled_cols]
y = df['price']

model = CatBoostRegressor(verbose=0)
model.fit(X, y)

