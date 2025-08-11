import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

# Load Zillow data
data = pd.read_csv('zillow.csv')

# Encode categorical column 'zipcode' as string
data['zipcode'] = data['zipcode'].astype(str)

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare features and target
features = ['bedrooms', 'bathrooms', 'sqft_living', 'zipcode']
X_train = train_data[features]
y_train = train_data['price']
X_test = test_data[features]
y_test = test_data['price']

# Train CatBoost model with categorical features specified
model = CatBoostRegressor(iterations=150, learning_rate=0.05, depth=8, verbose=0)
model.fit(X_train, y_train, cat_features=['zipcode'])

# Evaluate on test data
predictions = model.predict(X_test)

