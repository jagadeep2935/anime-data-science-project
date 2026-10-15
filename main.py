# ==========================================
# ANIME DATA SCIENCE PROJECT - DAY 1 SETUP
# ==========================================

import pandas as pd

# 1. Load the dataset
df = pd.read_csv('animes.csv')

print('Dataset Successfully Loaded')
print(f'Total Anime Records: {df.shape[0]}')
print(f'Total Features (Columns): {df.shape[1]}\n')

# 2. Display basic info
print('--- Column Data Types & Missing Values ---')
print(df.info())

# 3. Preview first few rows
print('\nFirst 3 Rows')
print(df.head(3))

# 4. Basic summary statistics
print('\nStatistical Summary (Numerical Columns)')
print(df.describe())

