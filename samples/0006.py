import pandas as pd
from catboost import CatBoostRegressor

df = pd.read_csv('zillow_sample_6.csv')
df = df[df['price'] > 0]
df['log_price'] = np.log1p(df['price'])

X = df[['bedrooms', 'bathrooms', 'sqft_living']]
y = df['log_price']

model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=4, verbose=0)
model.fit(X, y)

