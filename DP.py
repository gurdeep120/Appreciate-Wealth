import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv(r'C:\Users\Dell\OneDrive\Desktop\Apprenciate\data\fraudTrain.csv')
test_data = pd.read_csv(r'C:\Users\Dell\OneDrive\Desktop\Apprenciate\data\fraudTest.csv')

# Sample a subset of the data for quicker analysis (if needed)
# Uncomment the following line if you want to work with a sample
# train_data = train_data.sample(n=10000, random_state=1)

# 1. Basic Information and Summary
print("Dataset Summary:")
print(train_data.info())  # Check data types and non-null counts

# Check for missing values
print("\nMissing Values:")
print(train_data.isnull().sum())  # Check for any missing values

# 2. Analyze Distribution of Key Features

# Transaction Amount Distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_data['amt'], bins=50)
plt.title('Transaction Amount Distribution')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.show()

# Fraud vs Legitimate Transactions
plt.figure(figsize=(6, 4))
sns.countplot(x='is_fraud', data=train_data)
plt.title('Fraud vs Legitimate Transactions')
plt.xlabel('Is Fraud')
plt.ylabel('Count')
plt.show()

# 3. Time-Based Patterns
# Convert 'trans_date_trans_time' to datetime for time-based analysis
train_data['trans_date_trans_time'] = pd.to_datetime(train_data['trans_date_trans_time'])

# Extract hour and month from transaction time
train_data['hour'] = train_data['trans_date_trans_time'].dt.hour
train_data['month'] = train_data['trans_date_trans_time'].dt.month

# Fraud Transactions by Hour of the Day
plt.figure(figsize=(10, 6))
sns.countplot(x='hour', hue='is_fraud', data=train_data)
plt.title('Fraud Transactions by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Transaction Count')
plt.legend(title='Is Fraud', loc='upper right')
plt.show()

# Fraud Transactions by Month
plt.figure(figsize=(10, 6))
sns.countplot(x='month', hue='is_fraud', data=train_data)
plt.title('Fraud Transactions by Month')
plt.xlabel('Month')
plt.ylabel('Transaction Count')
plt.legend(title='Is Fraud', loc='upper right')
plt.show()

# 4. Correlation Matrix
# Select only numeric columns for correlation matrix
numeric_data = train_data.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
plt.figure(figsize=(12, 8))
correlation = numeric_data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

# 5. Check Merchant and Category for Fraud Patterns (Optional)
plt.figure(figsize=(10, 6))
top_merchants = train_data['merchant'].value_counts().index[:10]
sns.countplot(y='merchant', hue='is_fraud', data=train_data[train_data['merchant'].isin(top_merchants)], order=top_merchants)
plt.title('Top 10 Merchants with Fraud Transactions')
plt.xlabel('Transaction Count')
plt.ylabel('Merchant')
plt.show()

# Check if other categorical features like 'category' indicate fraud patterns (if present)
# You can explore similar to how you analyze 'merchant'

# Data Preprocessing
def data_preprocessing(df):
    # Handle missing or inconsistent data
    df.ffill(inplace=True)

    # Ensure 'dob' is in datetime format
    if df['dob'].dtype == 'object':
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    
    # Handle any remaining missing values in 'dob'
    df['dob'].fillna(pd.Timestamp.now(), inplace=True)

    # Feature engineering techniques
    # Deriving new features or aggregating information
    df['age'] = (pd.Timestamp.now() - df['dob']).dt.days / 365.25

# Apply data preprocessing to both datasets
data_preprocessing(train_data)
data_preprocessing(test_data)

# Check the preprocessed data
print("\nPreprocessed Training Data:")
print(train_data.head())
print("\nPreprocessed Test Data:")
print(test_data.head())
