import pandas as pd

# Load Zillow data
data = pd.read_csv('zillow.csv')

# Fill missing 'bathrooms' values with median
data['bathrooms'].fillna(data['bathrooms'].median(), inplace=True)

# Create a new feature 'price_per_sqft'
data['price_per_sqft'] = data['price'] / data['sqft_living']

# Filter out homes with unrealistic prices
data = data[(data['price'] > 50000) & (data['price'] < 2000000)]

