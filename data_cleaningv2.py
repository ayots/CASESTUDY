# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset with encoding fix
# Treat 'na', 'NA', 'Unknown', and empty strings as missing values
df_raw = pd.read_csv('./PhishingEmailData.csv', encoding='ISO-8859-1', na_values=['na', 'NA', 'Unknown', ''])

# Step 2: Clean column names
# Strip whitespace and standardize column names
df_raw.columns = df_raw.columns.str.strip().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)

# Step 3: Create a working copy of the dataset
df = df_raw.copy()

# Step 4: Visualise missing values BEFORE cleaning (excluding Email_Content)
missing_before = df.drop(columns=['Email_Content'], errors='ignore').isnull().sum()

plt.figure(figsize=(10, 5))
sns.barplot(x=missing_before.index, y=missing_before.values, palette='Reds_r')
plt.title("Missing Values Before Cleaning")
plt.ylabel("Count of Missing Entries")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Step 5: Clean and impute Sending_Date
if 'Sending_Date' in df.columns:
    # Convert to datetime format; invalid formats become NaT
    df['Sending_Date'] = pd.to_datetime(df['Sending_Date'], errors='coerce')
    # Fill missing dates with the most frequent valid date
    mode_date = df['Sending_Date'].mode(dropna=True)
    if not mode_date.empty:
        df['Sending_Date'] = df['Sending_Date'].fillna(mode_date[0])

# Step 6: Clean and impute Sending_Time and Day
for col in ['Sending_Time', 'Day']:
    if col in df.columns:
        df[col] = df[col].astype(str).replace(['nan', 'NaT', 'NA', 'na', '', 'Unknown'], pd.NA)
        mode_value = df[col].mode(dropna=True)
        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value[0])
        else:
            df[col] = df[col].fillna('Unknown')

# Step 7: Fill missing values in 'To' with 'Unknown'
if 'To' in df.columns:
    df['To'] = df['To'].fillna('Unknown')

# Step 8: Drop 'Sender_Name' column if it exists
if 'Sender_Name' in df.columns:
    df.drop(columns=['Sender_Name'], inplace=True)

# Step 9: Visualise missing values AFTER cleaning
missing_after = df.drop(columns=['Email_Content'], errors='ignore').isnull().sum()

plt.figure(figsize=(10, 5))
sns.barplot(x=missing_after.index, y=missing_after.values, palette='Greens')
plt.title("Missing Values After Cleaning")
plt.ylabel("Count of Missing Entries")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Step 10: Comparison chart of missing values before vs after
missing_comparison = pd.DataFrame({
    'Before Cleaning': missing_before,
    'After Cleaning': missing_after
})

missing_comparison.plot(kind='bar', figsize=(10, 5), color=['red', 'green'])
plt.title("Comparison of Missing Values Before and After Cleaning")
plt.ylabel("Count of Missing Entries")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Step 11: Save the cleaned dataset to a new CSV file
df.to_csv('Cleaned_PhishingEmailData.csv', index=False)

# Step 12: Preview the cleaned dataset
print("\nPreview of cleaned dataset:")
print(df.head())
