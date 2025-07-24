import pandas as pd
from catboost import Pool, CatBoostRegressor

df = pd.read_csv('zillow_sample_9.csv')
df['zipcode'] = df['zipcode'].astype(str)
cat_features = ['zipcode']

train_pool = Pool(data=df[['bedrooms', 'zipcode']], label=df['price'], cat_features=cat_features)

model = CatBoostRegressor(iterations=50, depth=3, verbose=0)
model.fit(train_pool)

