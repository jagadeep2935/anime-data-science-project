# Anime Dataset: Data Preprocessing and Exploratory Data Analysis (EDA)

# Import the required libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --------------------------------------------------
# LOAD THE DATASET
# --------------------------------------------------

# Load the anime dataset from the local CSV file
df = pd.read_csv('animes.csv')

# Display the first five records
print(df.head())

# Display the last five records
print(df.tail())

# Display dataset information, including columns, data types, and non-null counts
df.info()

# Display descriptive statistics for numerical columns
print(df.describe())

# --------------------------------------------------
# DATA PREPROCESSING AND TYPE CONVERSION
# --------------------------------------------------

# Remove leading and trailing spaces from genre values
df['genre'] = df['genre'].str.strip()

# Standardize genre capitalization for consistency
df['genre'] = df['genre'].str.title()

# Convert numerical columns to appropriate numeric data types
# Invalid values are converted to NaN for further handling
df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
df['members'] = pd.to_numeric(df['members'], errors='coerce')
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
df['ranked'] = pd.to_numeric(df['ranked'], errors='coerce')
df['score'] = pd.to_numeric(df['score'], errors='coerce')

# --------------------------------------------------
# DUPLICATE VALUE DETECTION AND REMOVAL
# --------------------------------------------------

# Count and display duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicate rows from the dataset
df.drop_duplicates(inplace=True)

# --------------------------------------------------
# MISSING VALUE ANALYSIS AND HANDLING
# --------------------------------------------------

# Display the number of missing values in each column before handling
print("\nMissing values before handling:")
print(df.isnull().sum())

# Replace missing synopsis values with 'Unknown'
df['synopsis'] = df['synopsis'].fillna('Unknown')

# Replace missing episode counts with the median episode count
df['episodes'] = df['episodes'].fillna(df['episodes'].median())

# Replace missing rankings with the median ranking
df['ranked'] = df['ranked'].fillna(df['ranked'].median())

# Replace missing scores with the median score
# Note: These imputed scores are estimates, not actual ratings
df['score'] = df['score'].fillna(df['score'].median())

# Replace missing image URLs with 'Unknown'
df['img_url'] = df['img_url'].fillna('Unknown')

# Display the number of remaining missing values after handling
print("\nMissing values after handling:")
print(df.isnull().sum())

# --------------------------------------------------
# CORRELATION AND COVARIANCE ANALYSIS
# --------------------------------------------------

# Calculate correlations between numerical columns
# Correlation measures the strength and direction of linear relationships
print(df.corr(numeric_only=True))

# Calculate covariance between numerical columns
# Covariance indicates whether two variables tend to vary together
print(df.cov(numeric_only=True))

# --------------------------------------------------
# SKEWNESS AND KURTOSIS ANALYSIS
# --------------------------------------------------

# Calculate score skewness to understand distribution asymmetry
print('\nSkewness of Scores:', df['score'].skew())

# Calculate score kurtosis to examine distribution tail characteristics
print('Kurtosis of Scores:', df['score'].kurt())

# --------------------------------------------------
# OUTLIER DETECTION USING THE IQR METHOD
# --------------------------------------------------

# Calculate the first and third quartiles of anime scores
Q1 = df['score'].quantile(0.25)
Q3 = df['score'].quantile(0.75)

# Calculate the interquartile range
IQR = Q3 - Q1

# Identify scores outside the 1.5 × IQR boundaries
score_outliers = df[
    (df['score'] < (Q1 - 1.5 * IQR)) |
    (df['score'] > (Q3 + 1.5 * IQR))
]

# Display the number of detected score outliers
print(f'Number of score outliers detected: {len(score_outliers)}')

# --------------------------------------------------
# VISUALIZATION: DISTRIBUTION OF ANIME SCORES
# --------------------------------------------------

# Plot a histogram with a KDE curve to visualize score distribution
plt.figure(figsize=(8, 4))

sns.histplot(
    df['score'].dropna(),
    bins=25,
    kde=True,
    color='purple',
    edgecolor='black'
)

plt.title('Distribution of Anime Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

# --------------------------------------------------
# VISUALIZATION: SCORE BOXPLOT
# --------------------------------------------------

# Use a boxplot to visualize score spread and potential outliers
plt.figure(figsize=(6, 4))

sns.boxplot(
    x=df['score'],
    color='skyblue'
)

plt.title('Boxplot of Anime Scores (Outlier Detection)')
plt.xlabel('Score')
plt.show()

# --------------------------------------------------
# VISUALIZATION: CORRELATION HEATMAP
# --------------------------------------------------

# Select relevant numerical features for correlation analysis
corr = df[
    ['score', 'episodes', 'members', 'popularity', 'ranked']
].corr(numeric_only=True)

# Visualize correlations using a heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# --------------------------------------------------
# VISUALIZATION: COMMUNITY MEMBERS VS. ANIME SCORE
# --------------------------------------------------

# Retain records with positive member counts and available scores
# Positive values are required for the logarithmic x-axis
plot_df = df[
    (df['members'] > 0) &
    df['score'].notna()
]

# Plot the relationship between community size and anime scores
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=plot_df,
    x='members',
    y='score',
    alpha=0.3,
    color='dodgerblue'
)

# Use a logarithmic scale to make a wide range of member counts easier to view
plt.title('Community Members vs. Anime Score')
plt.xlabel('Number of Members (Log Scale)')
plt.ylabel('Score')
plt.xscale('log')
plt.show()
