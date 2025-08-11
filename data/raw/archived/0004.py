import pandas as pd

# Load Zillow dataset
data = pd.read_csv('zillow.csv')

# Remove duplicates
data = data.drop_duplicates()

# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Create a new feature for house age
data['house_age'] = 2024 - data['year_built']

# Drop rows with missing target
data = data.dropna(subset=['price'])

