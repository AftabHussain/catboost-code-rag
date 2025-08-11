import pandas as pd

df = pd.read_csv('zillow_sample_10.csv')
df = df.drop(columns=['id', 'date', 'lat', 'long'])

df['bath_per_bed'] = df['bathrooms'] / df['bedrooms'].replace(0, 1)
df = df[df['bath_per_bed'] < 5]

