import pandas as pd

df = pd.read_csv('zillow_sample_12.csv')
df = df[df['floors'] <= 3]
df['room_density'] = df['sqft_living'] / (df['bedrooms'] + df['bathrooms'])

df = df.dropna(subset=['price', 'room_density'])

