import pandas as pd
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor

df = pd.read_csv('zillow_sample_15.csv')
X = df[['bedrooms', 'bathrooms', 'sqft_living']]
y = df['price']

model = CatBoostRegressor(iterations=50, verbose=0)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

