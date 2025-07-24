import pandas as pd

df = pd.read_csv('zillow_sample_7.csv')
df = df[df['sqft_living'] < 10000]
df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)

df = df.dropna(subset=['price', 'bed_bath_ratio'])

