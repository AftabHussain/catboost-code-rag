import pandas as pd
from catboost import CatBoostRegressor

# Load Zillow data
data = pd.read_csv('zillow.csv')

# Fill missing values for 'sqft_lot' with median
data['sqft_lot'].fillna(data['sqft_lot'].median(), inplace=True)

# Feature selection
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']
X = data[features]
y = data['price']

# Initialize and train CatBoost regressor
model = CatBoostRegressor(iterations=300, learning_rate=0.07, depth=5, verbose=0)
model.fit(X, y)

# Make predictions
preds = model.predict(X)

