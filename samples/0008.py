import pandas as pd
from catboost import CatBoostRegressor

df = pd.read_csv('zillow_sample_8.csv')
df['renovated'] = (df['yr_renovated'] > 0).astype(int)
df.fillna({'bathrooms': df['bathrooms'].mean()}, inplace=True)

X = df[['bedrooms', 'bathrooms', 'sqft_living', 'renovated']]
y = df['price']

model = CatBoostRegressor(verbose=0)
model.fit(X, y)

