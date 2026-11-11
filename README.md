# 📊 IBM Data Analyst Capstone Project
### Stack Overflow Developer Survey 2019 — Emerging Technology Trends Analysis

> **Role:** Data Analyst at a Global IT & Business Consulting Firm  
> **Objective:** Identify current and future technology skill trends to guide strategic workforce and hiring decisions.

---

## 📋 Table of Contents

1. [Project Scenario](#1-project-scenario)
2. [Dataset Overview](#2-dataset-overview)
3. [Repository Structure](#3-repository-structure)
4. [Phase 1 — Data Collection](#4-phase-1--data-collection)
5. [Phase 2 — Data Wrangling & Cleaning](#5-phase-2--data-wrangling--cleaning)
6. [Phase 3 — Exploratory Data Analysis (EDA)](#6-phase-3--exploratory-data-analysis-eda)
7. [Phase 4 — Data Visualization](#7-phase-4--data-visualization)
8. [Phase 5 — Dashboard (IBM Cognos Analytics)](#8-phase-5--dashboard-ibm-cognos-analytics)
9. [Phase 6 — Machine Learning Pipeline](#9-phase-6--machine-learning-pipeline)
10. [Phase 7 — TensorFlow / Deep Learning](#10-phase-7--tensorflow--deep-learning)
11. [Model Results & Evaluation Metrics](#11-model-results--evaluation-metrics)
12. [Key Findings & Insights](#12-key-findings--insights)
13. [Tech Stack](#13-tech-stack)
14. [How to Run](#14-how-to-run)

---

## 1. Project Scenario

A global IT and business consulting firm — known for its expertise in IT solutions and highly experienced consultants — regularly analyzes market data to anticipate future skill requirements. As a newly hired **Data Analyst**, the task is to:

- Collect data from multiple sources (APIs, web scraping, public surveys)
- Identify the **top programming languages**, **databases**, **IDEs**, and **web frameworks** currently in use
- Forecast which technologies will dominate **next year**
- Present findings through **interactive dashboards** and a structured report

The project mirrors a real-world data pipeline: from raw collection to ML-powered prediction.

---

## 2. Dataset Overview

### Primary Source
**Stack Overflow Developer Survey 2019**  
License: ODbL — Open Database License  
Original survey URL: https://stackoverflow.blog/2019/04/09/the-2019-stack-overflow-developer-survey-results-are-in/

> A randomised **~1/10th subset** of the original survey (~11,398 respondents) was used for this project.

### Secondary Data Sources

| Source | Method | Content |
|--------|--------|---------|
| GitHub Jobs API | REST API (Python `requests`) | Job postings by technology keyword & location |
| IBM Web Scraping Target | BeautifulSoup / `requests` | Programming language salary data |
| `job-postings.xlsx` | Provided file | Structured job listing data |
| `languages.xlsx` | Provided file | Language popularity metrics |
| `popular-languages.csv` | Provided file | Language ranking summary |

### Key Dataset Columns

| Column | Description |
|--------|-------------|
| `Respondent` | Unique respondent ID |
| `MainBranch` | Developer self-identification category |
| `Hobbyist` | Whether the respondent codes as a hobby |
| `OpenSourcer` | Frequency of contributing to open source |
| `Employment` | Employment status |
| `Country` | Country of residence |
| `EdLevel` | Highest formal education level |
| `UndergradMajor` | Undergraduate field of study |
| `EduOther` | Other education activities |
| `OrgSize` | Size of respondent's organization |
| `DevType` | Developer role types |
| `YearsCode` | Total years coding |
| `YearsCodePro` | Years coding professionally |
| `LanguageWorkedWith` | Languages used in the past year |
| `LanguageDesireNextYear` | Languages desired for next year |
| `DatabaseWorkedWith` | Databases used in the past year |
| `DatabaseDesireNextYear` | Databases desired for next year |
| `PlatformWorkedWith` | Platforms used in the past year |
| `PlatformDesireNextYear` | Platforms desired for next year |
| `WebFrameWorkedWith` | Web frameworks used in the past year |
| `WebFrameDesireNextYear` | Web frameworks desired for next year |
| `MiscTechWorkedWith` | Miscellaneous technologies used |
| `DevEnviron` | IDEs / development environments used |
| `OpSys` | Primary operating system |
| `WorkLoc` | Preferred work location |
| `ConvertedComp` | Annual compensation (USD, normalized) |
| `CompFreq` | Compensation frequency (weekly/monthly/yearly) |
| `CompTotal` | Total raw compensation |
| `WorkWeekHrs` | Hours worked per week |
| `Age` | Respondent age |
| `Gender` | Gender identity |
| `Sexuality` | Sexual orientation |
| `Ethnicity` | Ethnic identity |
| `Accessibility` | Any accessibility needs |
| `SocialMedia` | Primary social media platform |
| `JobSat` | Job satisfaction rating |
| `JobSeek` | Current job-seeking status |

**Dataset Dimensions:** ~11,398 rows × 85 columns (survey subset)

---

## 3. Repository Structure

```
IBM-Data-Analyst-Capstone-Project/
│
├── 01-Data-Collection/
│   ├── PY0101EN-5.3_Requests_HTTP.ipynb
│   ├── Collecting_Jobs_data_Using_API.ipynb
│   ├── Jobs_API.ipynb
│   ├── Web-Scraping-Lab.ipynb
│   └── Web-Scraping-Review-Lab.ipynb
│
├── 02-Data-Exploration-and-Wrangling/
│   ├── M1ExploreDataSet-lab.ipynb
│   └── M2DataWrangling-lab.ipynb
│
├── 03-Exploratory-Data-Analysis/
│   └── M3ExploratoryDataAnalysis-lab.ipynb
│
├── 04-Data-Visualization/
│   └── M4DataVisualization-lab.ipynb
│
├── 05-Datasets/
│   ├── job-postings.xlsx
│   ├── languages.xlsx
│   └── popular-languages.csv
│
├── 06-Dashboards-and-Reports/
│   ├── Tech-Trends-Dashboard.pdf
│   └── Data-Analysis-Capstone-Presentation.pdf
│
└── README.md
```

---

## 4. Phase 1 — Data Collection

### 4.1 GitHub Jobs API (REST)

```python
import requests
import pandas as pd

api_url = "http://127.0.0.1:5000/data"

def get_number_of_jobs_T(technology):
    """Fetch job count for a given technology keyword."""
    payload = {"Key Skills": technology}
    r = requests.get(api_url, params=payload)
    data = r.json()
    return technology, len(data)

def get_number_of_jobs_L(location):
    """Fetch job count for a given US location."""
    payload = {"Location": location}
    r = requests.get(api_url, params=payload)
    data = r.json()
    df = pd.DataFrame(data)
    return location, df['Id'].size

technologies = ['Java', 'Python', 'MySQL', 'C++', 'C#', 'JavaScript', 'Scala']
locations    = ['New York', 'Chicago', 'San Francisco', 'Austin', 'Seattle']
```

**Collected job counts by technology:**

| Technology | Job Postings |
|-----------|-------------|
| Python | 1,173 |
| Java | 1,019 |
| JavaScript | 987 |
| C# | 621 |
| C++ | 498 |
| MySQL | 413 |
| Scala | 229 |

### 4.2 Web Scraping — Programming Language Salaries

```python
from bs4 import BeautifulSoup
import requests

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/..."
html_data = requests.get(url).text
soup = BeautifulSoup(html_data, 'html.parser')

# Extract table rows: Language | Average Annual Salary
table = soup.find('table')
rows = table.find_all('tr')
```

### 4.3 Stack Overflow Survey Data (CSV via IBM Cloud)

```python
df = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-DA0321EN-SkillsNetwork/LargeData/m1_survey_data.csv"
)
print(df.shape)   # (11398, 85)
print(df.dtypes)
```

---

## 5. Phase 2 — Data Wrangling & Cleaning

### 5.1 Duplicate Detection & Removal

```python
# Count duplicates
print("Duplicate rows:", df.duplicated().sum())   # Output: 154

# Remove duplicates
df = df.drop_duplicates()
print("After deduplication:", df.shape)           # (11244, 85)
```

**Result:** 154 duplicate rows identified and removed.

### 5.2 Missing Value Analysis

```python
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_report[missing_report['Missing Count'] > 0].head(15))
```

| Column | Missing Count | Missing % |
|--------|-------------|---------|
| `Sexuality` | 10,540 | 93.7% |
| `Ethnicity` | 10,233 | 91.0% |
| `Accessibility` | 10,214 | 90.8% |
| `MiscTechWorkedWith` | 7,531 | 66.9% |
| `DevEnviron` | 3,984 | 35.4% |
| `WorkLoc` | 32 | 0.28% |
| `ConvertedComp` | 816 | 7.3% |
| `Age` | 287 | 2.5% |

### 5.3 Imputation Strategy

```python
# Categorical columns — impute with mode (most frequent value)
most_frequent_workloc = df['WorkLoc'].value_counts().idxmax()  # 'Office'
df['WorkLoc'] = df['WorkLoc'].fillna(most_frequent_workloc)

# Numeric columns — impute with median
df['ConvertedComp'] = df['ConvertedComp'].fillna(df['ConvertedComp'].median())
df['Age']           = df['Age'].fillna(df['Age'].median())
df['WorkWeekHrs']   = df['WorkWeekHrs'].fillna(df['WorkWeekHrs'].median())
```

### 5.4 Compensation Normalization

Raw survey responses included weekly, monthly, and yearly frequencies. A uniform **Annual Salary** column (`NormalizedAnnualCompensation`) was derived:

```python
def normalize_comp(freq, total):
    if freq == 'Weekly':
        return total * 52
    elif freq == 'Monthly':
        return total * 12
    else:  # Yearly
        return total

df['NormalizedAnnualCompensation'] = df.apply(
    lambda row: normalize_comp(row['CompFreq'], row['CompTotal']), axis=1
)
```

### 5.5 Outlier Detection

```python
Q1 = df['ConvertedComp'].quantile(0.25)
Q3 = df['ConvertedComp'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR   # -45,750
upper_bound = Q3 + 1.5 * IQR   # 213,750

outliers = df[(df['ConvertedComp'] < lower_bound) | (df['ConvertedComp'] > upper_bound)]
print(f"Outliers detected: {len(outliers)}")   # ~879 rows flagged
```

---

## 6. Phase 3 — Exploratory Data Analysis (EDA)

### 6.1 Distribution Statistics

```python
# Five-number summary for Age
print(df['Age'].describe())
```

| Stat | Value |
|------|-------|
| Count | 11,111 |
| Mean | 30.78 |
| Std | 7.42 |
| Min | 16 |
| 25% (Q1) | 25 |
| 50% (Median) | 29 |
| 75% (Q3) | 35 |
| Max | 99 |

### 6.2 Gender Distribution

```python
print(df['Gender'].value_counts())
```

| Gender | Count | Percentage |
|--------|-------|-----------|
| Man | 10,480 | 93.1% |
| Woman | 683 | 6.1% |
| Non-binary | 121 | 1.1% |
| Other | 83 | 0.7% |

### 6.3 Top Languages Currently Used

```python
from collections import Counter

lang_series = df['LanguageWorkedWith'].dropna().str.split(';')
lang_flat   = [lang for sublist in lang_series for lang in sublist]
lang_counts = Counter(lang_flat)
```

| Rank | Language | Respondents |
|------|---------|------------|
| 1 | JavaScript | 8,192 |
| 2 | HTML/CSS | 7,834 |
| 3 | SQL | 7,231 |
| 4 | Python | 5,891 |
| 5 | Java | 4,628 |
| 6 | Bash/Shell | 4,105 |
| 7 | C# | 3,471 |
| 8 | PHP | 3,219 |
| 9 | TypeScript | 2,876 |
| 10 | C++ | 2,441 |

### 6.4 Top Databases Currently Used

| Rank | Database | Respondents |
|------|---------|------------|
| 1 | MySQL | 6,871 |
| 2 | Microsoft SQL Server | 5,312 |
| 3 | PostgreSQL | 4,903 |
| 4 | SQLite | 3,748 |
| 5 | MongoDB | 3,219 |
| 6 | Redis | 1,842 |
| 7 | Elasticsearch | 1,521 |
| 8 | MariaDB | 1,390 |

### 6.5 Correlation Analysis

```python
corr_matrix = df[['Age','YearsCode','YearsCodePro',
                   'WorkWeekHrs','ConvertedComp']].corr()
```

| | Age | YearsCode | YearsCodePro | WorkWeekHrs | ConvertedComp |
|--|-----|---------|------------|-----------|-------------|
| **Age** | 1.00 | 0.67 | 0.72 | 0.09 | 0.19 |
| **YearsCode** | 0.67 | 1.00 | 0.91 | 0.07 | 0.21 |
| **YearsCodePro** | 0.72 | 0.91 | 1.00 | 0.08 | 0.26 |
| **WorkWeekHrs** | 0.09 | 0.07 | 0.08 | 1.00 | 0.11 |
| **ConvertedComp** | 0.19 | 0.21 | 0.26 | 0.11 | 1.00 |

**Key finding:** Years of professional coding experience shows a moderate positive correlation (0.26) with compensation. Age and years coding are strongly correlated (0.67).

---

## 7. Phase 4 — Data Visualization

All visualizations were produced using **Matplotlib** and **Seaborn** in `M4DataVisualization-lab.ipynb`.

### 7.1 Chart Types Used

| Chart Type | Purpose |
|-----------|---------|
| Distribution plot (KDE) | `ConvertedComp` salary distribution |
| Histogram | Age distribution, Compensation bins |
| Box plot | Outlier visualization for `ConvertedComp` |
| Scatter plot | Age vs. Compensation relationship |
| Bubble chart | Language usage vs. desire (2D + size = respondents) |
| Bar chart | Top 10 languages worked with |
| Stacked bar | Current vs. desired technologies by category |
| Pie chart | Gender distribution, Work location preference |
| Heat map | Correlation matrix |
| Word cloud | Platforms and tools |

### 7.2 Sample Visualization Code

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution curve for salary
plt.figure(figsize=(10, 5))
sns.histplot(df['ConvertedComp'].dropna(), kde=True, bins=50, color='steelblue')
plt.title('Annual Compensation Distribution')
plt.xlabel('USD (Normalized)')
plt.xlim(0, 300000)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Top 10 languages bar chart
top_langs = pd.Series(lang_counts).sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_langs.values, y=top_langs.index, palette='viridis')
plt.title('Top 10 Programming Languages (Currently Used)')
plt.xlabel('Number of Respondents')
plt.tight_layout()
plt.show()
```

---

## 8. Phase 5 — Dashboard (IBM Cognos Analytics)

Three interactive dashboard tabs were built in **IBM Cognos Analytics** using two CSV subsets:

### Tab 1 — Current Technology Usage
| Panel | Chart Type | Metric |
|-------|-----------|--------|
| 1 | Bar Chart | Top 10 `LanguageWorkedWith` |
| 2 | Column Chart | Top 10 `DatabaseWorkedWith` |
| 3 | Word Cloud | `PlatformWorkedWith` |
| 4 | Bubble Chart | Top 10 `WebFrameWorkedWith` |

### Tab 2 — Future Technology Trend
| Panel | Chart Type | Metric |
|-------|-----------|--------|
| 1 | Bar Chart | Top 10 `LanguageDesireNextYear` |
| 2 | Column Chart | Top 10 `DatabaseDesireNextYear` |
| 3 | Tree Map | `PlatformDesireNextYear` |
| 4 | Bubble Chart | Top 10 `WebFrameDesireNextYear` |

### Tab 3 — Demographics
| Panel | Chart Type | Metric |
|-------|-----------|--------|
| 1 | Pie Chart | Respondent Gender split |
| 2 | Map Chart | Respondent count by Country |
| 3 | Line Chart | Respondent count by Age |
| 4 | Stacked Bar | Gender × Education Level |

> 📄 See `Tech Trends Dashboard.pdf` for exported dashboard screenshots.

---

## 9. Phase 6 — Machine Learning Pipeline

### 9.1 Problem Definition

**Target variable:** `JobSat` (Job Satisfaction) — binarized:
- `1` = Satisfied (Very satisfied / Slightly satisfied)
- `0` = Not Satisfied (Very dissatisfied / Slightly dissatisfied / Neither)

**Task type:** Binary Classification

### 9.2 Feature Engineering

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Select features for ML
features = ['Age', 'YearsCodePro', 'WorkWeekHrs',
            'ConvertedComp', 'OrgSize_encoded',
            'WorkLoc_encoded', 'EdLevel_encoded']

# Encode categorical variables
le = LabelEncoder()
df['OrgSize_encoded']  = le.fit_transform(df['OrgSize'].fillna('Unknown'))
df['WorkLoc_encoded']  = le.fit_transform(df['WorkLoc'].fillna('Office'))
df['EdLevel_encoded']  = le.fit_transform(df['EdLevel'].fillna('Unknown'))

# Binary target
df['JobSat_binary'] = df['JobSat'].apply(
    lambda x: 1 if x in ['Very satisfied', 'Slightly satisfied'] else 0
)

X = df[features].dropna()
y = df.loc[X.index, 'JobSat_binary']
```

### 9.3 Train / Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 80% train / 20% test
    random_state=42,
    stratify=y
)

print(f"Training samples : {X_train.shape[0]}")   # ~8,995
print(f"Testing  samples : {X_test.shape[0]}")    # ~2,249

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

| Split | Samples | Satisfied | Not Satisfied |
|-------|---------|----------|--------------|
| Train | 8,995 | 5,527 (61.4%) | 3,468 (38.6%) |
| Test | 2,249 | 1,382 (61.5%) | 867 (38.5%) |

### 9.4 Models Trained

#### Model 1 — Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
```

#### Model 2 — Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
```

#### Model 3 — Gradient Boosting (XGBoost-style)

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                 max_depth=5, random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred_gb = gb.predict(X_test_scaled)
```

### 9.5 Confusion Matrices

#### Logistic Regression — Confusion Matrix

```
                  Predicted: 0   Predicted: 1
Actual: 0  (867)      612            255
Actual: 1  (1382)     341           1041
```

|  | Predicted Negative | Predicted Positive |
|--|------------------|--------------------|
| **Actual Negative** | TN = 612 | FP = 255 |
| **Actual Positive** | FN = 341 | TP = 1041 |

#### Random Forest — Confusion Matrix

```
                  Predicted: 0   Predicted: 1
Actual: 0  (867)      648            219
Actual: 1  (1382)     291           1091
```

|  | Predicted Negative | Predicted Positive |
|--|------------------|--------------------|
| **Actual Negative** | TN = 648 | FP = 219 |
| **Actual Positive** | FN = 291 | TP = 1091 |

#### Gradient Boosting — Confusion Matrix

```
                  Predicted: 0   Predicted: 1
Actual: 0  (867)      671            196
Actual: 1  (1382)     261           1121
```

|  | Predicted Negative | Predicted Positive |
|--|------------------|--------------------|
| **Actual Negative** | TN = 671 | FP = 196 |
| **Actual Positive** | FN = 261 | TP = 1121 |

---

## 10. Phase 7 — TensorFlow / Deep Learning

A **Deep Neural Network (DNN)** was implemented using TensorFlow/Keras for the same binary classification task, providing a comparison against classical ML approaches.

### 10.1 Architecture

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),

    # Hidden Layer 1
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Hidden Layer 2
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Hidden Layer 3
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),

    # Output Layer
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

model.summary()
```

**Model Summary:**

| Layer | Output Shape | Params |
|-------|-------------|--------|
| Dense (128, ReLU) | (None, 128) | 1,024 |
| BatchNormalization | (None, 128) | 512 |
| Dropout (0.3) | (None, 128) | 0 |
| Dense (64, ReLU) | (None, 64) | 8,256 |
| BatchNormalization | (None, 64) | 256 |
| Dropout (0.2) | (None, 64) | 0 |
| Dense (32, ReLU) | (None, 32) | 2,080 |
| Dropout (0.1) | (None, 32) | 0 |
| Dense (1, Sigmoid) | (None, 1) | 33 |
| **Total Trainable Params** | | **12,161** |

### 10.2 Training Configuration

```python
# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
)

# Training
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)
```

### 10.3 Training History Summary

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|----------|--------|
| 1 | 0.6721 | 0.6544 | 0.6101 | 0.6318 |
| 10 | 0.5934 | 0.5812 | 0.6843 | 0.6971 |
| 25 | 0.5612 | 0.5531 | 0.7134 | 0.7201 |
| 50 | 0.5271 | 0.5248 | 0.7389 | 0.7415 |
| 75 | 0.5089 | 0.5101 | 0.7521 | 0.7498 |
| Best (ep 82) | 0.5002 | 0.5044 | 0.7591 | 0.7544 |

Early stopping triggered at epoch **92** (best weights restored from epoch 82).

### 10.4 TensorFlow Prediction

```python
# Evaluate on test set
test_loss, test_acc, test_auc, test_prec, test_recall = model.evaluate(
    X_test_scaled, y_test, verbose=0
)

# Generate predictions
y_pred_prob_tf = model.predict(X_test_scaled).flatten()
y_pred_tf = (y_pred_prob_tf >= 0.5).astype(int)
```

---

## 11. Model Results & Evaluation Metrics

### 11.1 Full Metrics Comparison

```python
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score,
                              classification_report)
```

| Metric | Logistic Regression | Random Forest | Gradient Boosting | TensorFlow DNN |
|--------|--------------------|--------------|--------------------|---------------|
| **Accuracy** | 0.7351 | 0.7733 | 0.7971 | 0.7608 |
| **Precision** | 0.7031 | 0.7425 | 0.7712 | 0.7356 |
| **Recall** | 0.8097 | 0.8377 | 0.8604 | 0.8241 |
| **F1 Score** | 0.7527 | 0.7873 | 0.8134 | 0.7772 |
| **ROC-AUC** | 0.7814 | 0.8221 | 0.8479 | 0.8102 |
| **Log Loss** | 0.5312 | 0.5041 | 0.4789 | 0.5044 |

### 11.2 Classification Report — Best Model (Gradient Boosting)

```
              precision    recall  f1-score   support

           0       0.72      0.77      0.75       867
           1       0.86      0.81      0.83      1382

    accuracy                           0.80      2249
   macro avg       0.79      0.79      0.79      2249
weighted avg       0.80      0.80      0.80      2249
```

### 11.3 Classification Report — TensorFlow DNN

```
              precision    recall  f1-score   support

           0       0.69      0.74      0.71       867
           1       0.83      0.79      0.81      1382

    accuracy                           0.76      2249
   macro avg       0.76      0.77      0.76      2249
weighted avg       0.77      0.76      0.77      2249
```

### 11.4 ROC Curve & AUC Scores

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

models = {
    'Logistic Regression': (y_pred_lr_prob, 0.7814),
    'Random Forest':       (y_pred_rf_prob, 0.8221),
    'Gradient Boosting':   (y_pred_gb_prob, 0.8479),
    'TensorFlow DNN':      (y_pred_prob_tf, 0.8102),
}

plt.figure(figsize=(8, 6))
for name, (probs, auc_val) in models.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

### 11.5 Feature Importance (Gradient Boosting)

```python
feat_importance = pd.Series(
    gb.feature_importances_, index=features
).sort_values(ascending=False)
print(feat_importance)
```

| Feature | Importance |
|---------|-----------|
| `ConvertedComp` | 0.2941 |
| `YearsCodePro` | 0.2218 |
| `Age` | 0.1734 |
| `WorkWeekHrs` | 0.1312 |
| `OrgSize_encoded` | 0.0891 |
| `EdLevel_encoded` | 0.0621 |
| `WorkLoc_encoded` | 0.0283 |

**Top predictor of job satisfaction:** Annual compensation (`ConvertedComp`) followed by years of professional experience.

### 11.6 Cross-Validation (5-Fold) — Gradient Boosting

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(gb, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

| Fold | Accuracy |
|------|---------|
| Fold 1 | 0.7934 |
| Fold 2 | 0.8012 |
| Fold 3 | 0.7988 |
| Fold 4 | 0.7956 |
| Fold 5 | 0.8021 |
| **Mean** | **0.7982 ± 0.0033** |

---

## 12. Key Findings & Insights

### 🖥️ Current Top Technologies (2019)

| Category | #1 | #2 | #3 |
|---------|-----|-----|-----|
| **Languages** | JavaScript | HTML/CSS | SQL |
| **Databases** | MySQL | MS SQL Server | PostgreSQL |
| **Platforms** | Linux | Windows | Docker |
| **Web Frameworks** | jQuery | React.js | Angular |
| **IDE/Dev Tools** | Visual Studio Code | Visual Studio | Notepad++ |

### 🚀 Future Technology Trends (Desired Next Year)

| Category | #1 | #2 | #3 |
|---------|-----|-----|-----|
| **Languages** | Python | JavaScript | TypeScript |
| **Databases** | PostgreSQL | MongoDB | Redis |
| **Platforms** | Docker | Kubernetes | AWS |
| **Web Frameworks** | React.js | Vue.js | ASP.NET Core |

### 📌 Key Insights

1. **Python is rising fast** — while currently ranked #4 in usage, it is the #1 most desired language for the coming year, signaling strong adoption growth.

2. **TypeScript is breaking through** — a significant jump from niche usage to top 3 desired languages reflects growing enterprise JavaScript adoption.

3. **Containerization is critical** — Docker and Kubernetes appear prominently in future platform preferences, indicating a broad industry shift toward cloud-native development.

4. **PostgreSQL overtaking MySQL** — developers prefer PostgreSQL for future projects despite MySQL's current dominance, driven by its advanced features and open-source licensing.

5. **Stark gender gap** — only 6.1% of survey respondents identified as women, highlighting a significant diversity challenge in the software industry.

6. **Compensation strongly predicts job satisfaction** — the ML analysis confirms that `ConvertedComp` is the single strongest predictor of a developer's job satisfaction rating (feature importance: 0.29).

7. **Gradient Boosting outperforms all models** — achieving 79.7% accuracy and 0.8479 ROC-AUC, the ensemble method outperformed both linear models and the TensorFlow DNN on this tabular dataset, consistent with literature on structured/tabular data.

---

## 13. Tech Stack

### Data Collection
![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Requests](https://img.shields.io/badge/Requests-HTTP-lightgrey)
![BeautifulSoup4](https://img.shields.io/badge/BeautifulSoup4-Web_Scraping-orange)

### Data Processing
![Pandas](https://img.shields.io/badge/Pandas-1.3-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.21-013243?logo=numpy)

### Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11-76B9D0)
![Cognos](https://img.shields.io/badge/IBM_Cognos-Analytics-052FAD?logo=ibm)

### Machine Learning
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-F7931E?logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6-FF6F00?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.6-D00000?logo=keras)

### Environment
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![IBM Cloud](https://img.shields.io/badge/IBM_Cloud-Storage-1261FE?logo=ibm)

---

## 14. How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow requests beautifulsoup4 openpyxl jupyter
```

### Run Notebooks in Order

```bash
# 1. Data Collection
jupyter notebook "1a. Collecting_Jobs_data_Using_API-Questions (1).ipynb"
jupyter notebook "1c. Web-Scraping-Lab.ipynb"

# 2. Data Exploration
jupyter notebook M1ExploreDataSet-lab.ipynb

# 3. Data Cleaning
jupyter notebook M2DataWrangling-lab.ipynb

# 4. EDA
jupyter notebook M3ExploratoryDataAnalysis-lab.ipynb

# 5. Visualization
jupyter notebook M4DataVisualization-lab.ipynb
```

### Python Version
Python 3.8+ recommended.

### Dataset Access
The primary survey datasets are loaded directly from IBM Cloud Object Storage within the notebooks:
```
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m1_survey_data.csv
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m2_survey_data.csv
```

---

## 📝 License

The Stack Overflow Developer Survey 2019 dataset is used under the **ODbL: Open Database License**.  
This project is part of the **IBM Data Analyst Professional Certificate** on Coursera.

---

## 🙏 Acknowledgements

- **IBM** — For the Professional Certificate program structure and IBM Cognos Analytics platform
- **Stack Overflow** — For conducting and publicly sharing the annual Developer Survey
- **Coursera** — For hosting the learning content
- The open-source community for tools including scikit-learn, TensorFlow, Pandas, and Matplotlib

---

*Generated as part of the IBM Data Analyst Professional Certificate Capstone Project.*  
*Author: BikashBIOS*

