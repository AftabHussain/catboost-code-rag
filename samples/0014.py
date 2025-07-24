import pandas as pd

df = pd.read_csv('zillow_sample_14.csv')
df['price_sqft'] = df['price'] / df['sqft_living']
grouped = df.groupby('zipcode')['price_sqft'].median().reset_index()

