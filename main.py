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

# Clean and standardize genre values
df['genre'] = (
    df['genre']
    .str.strip()
    .str.strip("[]")
    .str.replace("'", "")
    .str.replace('"', '')
    .str.strip()
    .str.title()
)

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

# --------------------------------------------------
# FIVE OBJECTIVES OF THE ANIME DATA SCIENCE PROJECT
# --------------------------------------------------

# Objective 1: Analyze Anime Ratings
highest_rated = df[['title', 'score']].sort_values('score', ascending=False).head(10)
lowest_rated = df[['title', 'score']].sort_values('score', ascending=True).head(10)
print("\nOBJECTIVE 1: Anime Rating Analysis")
print(highest_rated)
print(lowest_rated)

# Objective 2: Study Popularity and Community Engagement
members_score_corr = df['members'].corr(df['score'])
popularity_score_corr = df['popularity'].corr(df['score'])
print("\nOBJECTIVE 2: Popularity and Community Engagement")
print("Members vs Score:", members_score_corr)
print("Popularity vs Score:", popularity_score_corr)

# Objective 3: Investigate Genre-Based Rating Patterns
genre_df = df.copy()
genre_df['genre'] = genre_df['genre'].str.split(', ')
genre_df = genre_df.explode('genre')
genre_scores = genre_df.groupby('genre')['score'].mean()
top_genres_by_score = genre_scores.sort_values(ascending=False).head(10)
print("\nOBJECTIVE 3: Genre-Based Rating Analysis")
print(top_genres_by_score)

# Objective 4: Examine Episode Count and Ratings
episode_score_corr = df['episodes'].corr(df['score'])
df['episode_group'] = pd.cut(
    df['episodes'],
    bins=[-1, 1, 12, 26, 100, float('inf')],
    labels=['1 Episode', '2-12', '13-26', '27-100', '100+']
)
episode_group_scores = df.groupby(
    'episode_group', observed=False
)['score'].mean()
print("\nOBJECTIVE 4: Episode Count and Rating Analysis")
print("Episodes vs Score:", episode_score_corr)
print(episode_group_scores)

# Objective 5: Identify the Most Common Anime Genres
genre_counts = df['genre'].str.split(', ').explode().value_counts()
top_genres = genre_counts.head(10)
print("\nOBJECTIVE 5: Most Common Anime Genres")
print(top_genres)
top_genres.plot(kind='bar', figsize=(8, 5))
plt.title('Top 10 Most Common Anime Genres')
plt.xlabel('Genre')
plt.ylabel('Number of Anime')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

