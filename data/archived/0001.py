import pandas as pd
from catboost import CatBoostRegressor

# Load Zillow dataset
data = pd.read_csv('zillow.csv')

# Drop rows with missing target values
data = data.dropna(subset=['price'])

# Select features and target
features = ['bedrooms', 'bathrooms', 'sqft_living']
X = data[features]
y = data['price']

# Train CatBoost model
model = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, verbose=0)
model.fit(X, y)

# Predict prices
predictions = model.predict(X)

