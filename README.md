# 🎌 Anime Data Analysis & Rating Classification

An end-to-end Data Science project that explores anime ratings, popularity, community engagement, genre trends, and episode counts using Python. The project combines **data preprocessing, exploratory data analysis (EDA), data visualization, and machine learning classification** to investigate patterns in anime data and predict rating categories.

## 📌 Project Overview

Anime datasets contain valuable information about audience engagement, ratings, genres, and episode counts. This project analyzes these attributes to uncover meaningful patterns and evaluates four machine learning models to classify anime into **Low, Medium, and High rating categories**.

The project follows a complete data analysis workflow, from cleaning raw data to interpreting model evaluation results.

## 🎯 Project Objectives

1. **Anime Rating Analysis** — Identify the highest-rated and lowest-rated anime.
2. **Popularity and Community Engagement** — Examine relationships between scores, community members, and popularity rankings.
3. **Genre-Based Rating Analysis** — Identify genres with the highest average scores.
4. **Episode Count and Rating Analysis** — Investigate whether episode counts are associated with anime ratings.
5. **Most Common Anime Genres** — Determine which genres appear most frequently.

## 📂 Dataset

- **Records:** 19,311 rows in the original dataset before duplicate removal.
- **Features:** 12 columns in the original dataset.
- **Format:** CSV
- **Attributes:** Anime titles, genres, episodes, members, popularity, rankings, scores, and additional metadata.

*Add the original dataset source and link here, if available.*

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data analysis and model development |
| Pandas | Data manipulation and preprocessing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Classification, feature scaling, and model evaluation |

## 🧹 Data Preprocessing

- Standardized genre text by trimming whitespace and normalizing capitalization.
- Converted numerical columns to appropriate numeric data types.
- Detected and removed duplicate rows.
- Identified missing values in the dataset.
- Handled missing synopsis and image URL values.
- Imputed missing episode counts, rankings, and scores using median values.
- Verified missing-value counts after preprocessing.

**Data quality note:** Median-imputed scores are estimates, not actual anime ratings. This limitation should be considered when interpreting the classification results.

## 📊 Exploratory Data Analysis

### Visualizations

- **Anime Score Distribution:** Histogram with KDE curve.
- **Score Boxplot:** Identifies potential score outliers using the IQR method.
- **Correlation Heatmap:** Displays relationships among numerical features.
- **Community Members vs. Anime Score:** Scatter plot using a logarithmic scale for member counts.
- **Most Common Anime Genres:** Bar chart showing the ten most frequent genres.

### Statistical Analysis

- Correlation and covariance
- Score skewness and kurtosis
- Interquartile range (IQR) outlier detection
- Average scores by genre and episode count

## 🔍 Key Findings

- **Highest-rated anime:** *Fullmetal Alchemist: Brotherhood* — 9.23.
- **Lowest-rated anime:** *Phantomi: Mini Anime* — 1.25.
- **Members vs. Score:** Positive correlation of approximately **0.391**.
- **Popularity vs. Score:** Negative correlation of approximately **−0.656**. Lower numerical popularity ranks generally indicate greater popularity.
- **Episodes vs. Score:** Weak positive correlation of approximately **0.062**, indicating almost no linear relationship.
- **Top genre by average score:** Mystery — approximately 7.08.
- **Most common genre:** Comedy — 5,683 occurrences in the exploded genre data.

These findings describe associations in the dataset and do not establish causation.

## 🤖 Machine Learning Models

Anime scores are grouped into three rating categories:

| Category | Score interval |
|---|---|
| Low | Greater than 0 and up to 6 |
| Medium | Greater than 6 and up to 8 |
| High | Greater than 8 and up to 10 |

**Input features:**
- Episodes
- Members
- Popularity

The dataset is split into training and testing sets using a 70:30 ratio. Features are standardized using `StandardScaler`.

### Models Implemented

1. **Logistic Regression** — Classification baseline.
2. **K-Nearest Neighbors (KNN)** — Classifies anime based on nearby observations.
3. **Gaussian Naive Bayes** — Uses a probabilistic classification approach.
4. **Decision Tree Classifier** — Learns decision rules from the input features.

## 📈 Model Performance

The following results were obtained from the current test set:

| Model | Accuracy |
|---|---:|
| K-Nearest Neighbors (KNN) | **74.69%** |
| Logistic Regression | 72.55% |
| Decision Tree | 66.85% |
| Gaussian Naive Bayes | 64.92% |

**Best result by accuracy:** KNN achieved 74.69% accuracy in the current experiment.

The models were evaluated using:
- Accuracy
- Confusion matrix
- Precision
- Recall
- F1-score

> **Important limitation:** The High category has substantially fewer test samples than the Low and Medium categories. KNN achieved only 17% recall for the High category, meaning it misses many genuinely high-rated anime. Its overall accuracy should therefore be interpreted alongside the other evaluation metrics.

## 🌳 Decision Tree Visualization

The project generates a Decision Tree diagram to visualize learned classification rules. The visualization is limited to a depth of three levels for readability.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install the dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### 3. Add the dataset

Place the anime CSV file in the project directory, or update the dataset path in `main.py` to match its location.

### 4. Run the project

```bash
python main.py
```

The script prints dataset summaries, preprocessing results, statistical findings, objective-wise results, and model evaluation reports. It also generates EDA charts, a genre bar chart, and a Decision Tree visualization.

## 📁 Repository Structure

```text
Anime-Data-Analysis/
├── animes.csv
├── main.py
└── README.md
```

*Adjust filenames and structure to match the actual repository. Only upload the dataset if its license permits redistribution.*

## ⚠️ Limitations and Future Improvements

- Evaluate KNN values using cross-validation or a separate validation set.
- Address class imbalance and improve detection of the High rating category.
- Evaluate the effect of imputed scores by training on records with genuine ratings.
- Explore additional relevant features and compare alternative classification approaches.
- Save visualizations and add a detailed comparison of model performance.

## 🎓 Learning Outcomes

- Data cleaning and preprocessing with Pandas.
- Exploratory data analysis and statistical interpretation.
- Data visualization using Matplotlib and Seaborn.
- Feature scaling and train-test splitting.
- Implementing classification algorithms with Scikit-learn.
- Evaluating models and interpreting their limitations.

## 👨‍💻 Author

**Jagadeep Jaladi**

Computer Science and Engineering | Data Science

[LinkedIn](https://www.linkedin.com/in/jagadeepjaladi/)

---

*This project was developed as part of my Data Science learning journey to apply data analysis, visualization, and machine learning techniques to an anime dataset.*
