import pandas as pd
from catboost import CatBoostRegressor

df = pd.read_csv('zillow_sample_13.csv')
df = df[df['view'] > 0]
df['has_view'] = (df['view'] > 0).astype(int)

X = df[['sqft_living', 'has_view']]
y = df['price']

model = CatBoostRegressor(iterations=75, learning_rate=0.2, verbose=0)
model.fit(X, y)

