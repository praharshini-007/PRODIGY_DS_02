
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset (replace with the path to your local CSV if needed)
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Display first few rows
print(df.head())

# -----------------------------
# 1. DATA CLEANING
# -----------------------------

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Fill missing 'Age' with median, 'Embarked' with mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Verify cleaning
print("\nMissing values after cleaning:\n", df.isnull().sum())

# -----------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------

# Basic Info
print("\nDataset Info:")
print(df.info())

# Survival count
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.show()

# Survival by Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

# Age distribution
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()

# Boxplot: Age vs Survived
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age vs Survival')
plt.show()

# Heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot (optional for detailed variable relationships)
# sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']], hue='Survived')

# -----------------------------
# 3. CONCLUSION EXAMPLES:
# -----------------------------
# - Females had higher survival rates than males.
# - Higher class passengers (1st class) had higher survival chances.
# - Children (younger passengers) had a better chance of survival.