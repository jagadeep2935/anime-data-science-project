# Anime Dataset: Data Preprocessing and Exploratory Data Analysis (EDA)

# Import the required libraries
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# LOAD THE DATASET
# Load the anime dataset from the local CSV file
df = pd.read_csv('C:/Users/jalad/Downloads/archive (4)/animes.csv')

# Display the first five records
print(df.head(5))

# Display the last five records
print(df.tail(5))

# Display dataset information, including columns, data types, and non-null counts
df.info()

# Display descriptive statistics for numerical columns
print(df.describe())

# Removing the unwanted columns
df.drop(columns=['img_url', 'synopsis'], inplace=True)

# DATA PREPROCESSING AND TYPE CONVERSION

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
df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
df['members'] = pd.to_numeric(df['members'], errors='coerce')
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
df['ranked'] = pd.to_numeric(df['ranked'], errors='coerce')
df['score'] = pd.to_numeric(df['score'], errors='coerce')

# DUPLICATE VALUE DETECTION AND REMOVAL

# Count and display duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicate rows from the dataset
df.drop_duplicates(inplace=True)

# MISSING VALUE ANALYSIS AND HANDLING

# Display the number of missing values in each column before handling
print("\nMissing values before handling:")
print(df.isnull().sum())

# Replace missing episode counts with the median episode count
df['episodes'] = df['episodes'].fillna(df['episodes'].median())

# Replace missing rankings with the median ranking
df['ranked'] = df['ranked'].fillna(df['ranked'].median())

# Replace missing scores with the median score
# Note: These imputed scores are estimates, not actual ratings
df['score'] = df['score'].fillna(df['score'].median())

# Display the number of remaining missing values after handling
print("\nMissing values after handling:")
print(df.isnull().sum())

# CORRELATION AND COVARIANCE ANALYSIS

# Calculate correlations between numerical columns
# Correlation measures the strength and direction of linear relationships
print(df.corr(numeric_only=True))

# Calculate covariance between numerical columns
# Covariance indicates whether two variables tend to vary together
print(df.cov(numeric_only=True))

# SKEWNESS AND KURTOSIS ANALYSIS

# Calculate score skewness to understand distribution asymmetry
print('\nSkewness of Scores:', df['score'].skew())

# Calculate score kurtosis to examine distribution tail characteristics
print('Kurtosis of Scores:', df['score'].kurt())

# OUTLIER DETECTION USING THE IQR METHOD

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

# VISUALIZATION: DISTRIBUTION OF ANIME SCORES

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

# VISUALIZATION: SCORE BOXPLOT

# Use a boxplot to visualize score spread and potential outliers
plt.figure(figsize=(6, 4))

sns.boxplot(
    x=df['score'],
    color='skyblue'
)

plt.title('Boxplot of Anime Scores (Outlier Detection)')
plt.xlabel('Score')
plt.show()

# VISUALIZATION: CORRELATION HEATMAP

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

# VISUALIZATION: COMMUNITY MEMBERS VS. ANIME SCORE

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

# FIVE OBJECTIVES OF THE ANIME DATA SCIENCE PROJECT

# Objective 1: Analyze Anime Ratings
highest_rated = df[['title', 'score']].sort_values('score', ascending=False).head(10)
lowest_rated = df[['title', 'score']].sort_values('score', ascending=True).head(10)
print("\nOBJECTIVE 1: Anime Rating Analysis")
print("\nHighest Rated Anime: \n")
print(highest_rated)
print("\nLowest Rated Anime: \n")
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

# Create rating categories
df['rating_category'] = pd.cut(
    df['score'],
    bins=[0, 6, 8, 10],
    labels=['Low', 'Medium', 'High']
)

# Select input features and target variable
X = df[['episodes', 'members', 'popularity']]
Y = df['rating_category']
# Split data into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42,stratify=Y
)
# Standardize the features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# CLASSIFICATION MODELS

# CLASSIFICATION MODEL - 1 : Logistic Regression
# Create and train the Logistic Regression model
model1 = LogisticRegression(max_iter=1000,class_weight='balanced')
model1.fit(X_train, Y_train)
# Predict rating categories
y_pred1 = model1.predict(X_test)
# Evaluate the model
print("\nEvaluation parameters for Logistic regression model")
print("Accuracy:", accuracy_score(Y_test, y_pred1))
print("Precision:", precision_score(Y_test, y_pred1, average='macro'))
print("Recall:", recall_score(Y_test, y_pred1, average='macro'))
print("F1 Score:", f1_score(Y_test, y_pred1, average='macro'))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred1))
print("\nClassification Report:")
print(classification_report(Y_test, y_pred1))

# CLASSIFICATION MODEL - 2 : KNN 
# Create and train the KNN model
k = round(math.sqrt(len(X_train)))
print("Recommended K:", k)
model2 = KNeighborsClassifier(n_neighbors=k)
model2.fit(X_train, Y_train)
# Predict rating categories
y_pred2 = model2.predict(X_test)
# Evaluate the model
print("\nEvaluation parameters for KNN model")
print("Precision:", precision_score(Y_test, y_pred2, average='macro'))
print("Recall:", recall_score(Y_test, y_pred2, average='macro'))
print("F1 Score:", f1_score(Y_test, y_pred2, average='macro'))
print("Accuracy:", accuracy_score(Y_test, y_pred2))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred2))
print("\nClassification Report:")
print(classification_report(Y_test, y_pred2))

# CLASSIFICATION MODEL - 3 : Support Vector Machine
# Create and train the Support Vector Machine
model3 = SVC(kernel='linear',class_weight='balanced')
model3.fit(X_train, Y_train)
# Predict rating categories
y_pred3 = model3.predict(X_test)
# Evaluate the model
print("\nEvaluation parameters for Support Vector Machine")
print("Accuracy:", accuracy_score(Y_test, y_pred3))
print("Precision:", precision_score(Y_test, y_pred3, average='macro'))
print("Recall:", recall_score(Y_test, y_pred3, average='macro'))
print("F1 Score:", f1_score(Y_test, y_pred3, average='macro'))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred3))
print("\nClassification Report:")
print(classification_report(Y_test, y_pred3))

# CLASSIFICATION MODEL - 4 : Random Forest
# Create and train the Random Forest model
model4 = RandomForestClassifier(random_state=42,class_weight='balanced')
model4.fit(X_train, Y_train)
# Predict rating categories
y_pred4 = model4.predict(X_test)
# Evaluate the model
print("\nEvaluation parameters for Random Forest model")
print("Accuracy:", accuracy_score(Y_test, y_pred4))
print("Precision:", precision_score(Y_test, y_pred4, average='macro'))
print("Recall:", recall_score(Y_test, y_pred4, average='macro'))
print("F1 Score:", f1_score(Y_test, y_pred4, average='macro'))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred4))
print("\nClassification Report:")
print(classification_report(Y_test, y_pred4))

# MODEL ACCURACY COMPARISON

models = ['Logistic Regression', 'KNN','Support Vector Machine', 'Random Forest']

print("\nMODEL 1: Logistic Regression")
print("Accuracy:", accuracy_score(Y_test, y_pred1))
print("Precision:", precision_score(Y_test, y_pred1, average='macro'))
print("Recall:", recall_score(Y_test, y_pred1, average='macro'))
print("F1 Score:", f1_score(Y_test, y_pred1, average='macro'))

print("\nMODEL 2: KNN")
print("Accuracy:", accuracy_score(Y_test, y_pred2))
print("Precision:", precision_score(Y_test, y_pred2, average='macro'))
print("Recall:", recall_score(Y_test, y_pred2, average='macro'))
print("F1 Score:", f1_score(Y_test, y_pred2, average='macro'))

print("\nMODEL 3: Support Vector Machine")
print("Accuracy:", accuracy_score(Y_test, y_pred3))
print("Precision:", precision_score(Y_test, y_pred3, average='macro'))
print("Recall:", recall_score(Y_test, y_pred3, average='macro'))
print("F1 Score:", f1_score(Y_test, y_pred3, average='macro'))

print("\nMODEL 4: Random Forest")
print("Accuracy:", accuracy_score(Y_test, y_pred4))
print("Precision:", precision_score(Y_test, y_pred4, average='macro'))
print("Recall:", recall_score(Y_test, y_pred4, average='macro'))
print("F1 Score:", f1_score(Y_test, y_pred4, average='macro'))

# Best fit model
print("\nConcluding the best fit model")
print(
    "The linear Support Vector Machine achieved the highest macro F1 score (60.40%) with a recall of 71.07%, "
    "accuracy of 68.72% and precision of 57.90%. KNN achieved the highest accuracy (74.47%) and precision (68.21%), "
    "but its recall (55.75%) and macro F1 (57.30%) were lower. Logistic Regression (F1 57.39%) and "
    "Random Forest (F1 57.16%) performed similarly to KNN. Since the classes are imbalanced, macro F1 "
    "is a more reliable measure than accuracy, so SVM is selected as the best-fit model."
)

# KEY FINDINGS
print("\nKEY FINDINGS")
print(
    "1. Popularity rank (-0.66) and members (0.39) are the strongest predictors of score; episode count is weak (0.06). "
    "2. Mystery, Thriller and Shounen have the highest average scores, and anime with 13-26 episodes score highest (6.82). "
    "3. Comedy, Action and Fantasy are the most common genres. "
    "4. High-rated anime are rare (about 3.7% of the data), so accuracy alone is misleading. "
    "5. Class balancing raised High-class recall from 0.14 to 0.70-0.78 for the SVM and Logistic Regression models, but lowered accuracy. "
    "6. SVM gave the best macro F1 (60.40%), while KNN gave the best accuracy (74.47%)."
)

# FINAL CONCLUSION
print("\nFINAL CONCLUSION")
print(
    "Four models were used to classify anime into Low, Medium and High ratings. The linear SVM performed best "
    "overall with the highest macro F1 score (60.40%) and the best balance on High-rated anime (High-class F1 0.40), "
    "while KNN had the highest accuracy (74.47%). "
    "Performance is moderate because only three features were used and the classes are imbalanced."
)

print("\nRECOMMENDATIONS")
print("1. Anime with a large community and better popularity rank tend to score higher.")
print("2. Mystery, Thriller and Shounen genres are rated highest; 13-26 episode series score best.")
print("3. Use the linear SVM to predict rating categories, especially to find potential top-rated anime.")
print("4. Add features such as genre, studio and release year to improve results.")

print("\nLIMITATION")
print("The 404 missing scores were filled with the median score, which may inflate the Medium class.")
