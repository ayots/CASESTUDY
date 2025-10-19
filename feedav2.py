# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np

# Step 1: Load the dataset with encoding fix to avoid Unicode errors
df = pd.read_csv('./PhishingEmailData.csv', encoding='ISO-8859-1')

# Step 2: Clean column names to remove any leading/trailing whitespace
df.columns = df.columns.str.strip()

# Step 3: Manually add a label column for classification
# 0 = legitimate email, 1 = phishing email
# For demonstration, we randomly assign labels. Replace this with real labels if available.
np.random.seed(42)  # Ensures reproducibility
df['Label'] = np.random.choice([0, 1], size=len(df))

# Step 4: Define a function to clean email content
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r"http\\S+", "link", text)  # Replace URLs with 'link'
    text = re.sub(r"\\S+@\\S+", "email", text)  # Replace email addresses with 'email'
    text = re.sub(r"[^a-z\\s]", "", text)  # Remove non-alphabetic characters
    return text

# Step 5: Apply cleaning and extract metadata features
df['clean_content'] = df['Email_Content'].apply(clean_text)

# Calculate subject length
df['subject_length'] = df['Email_Subject'].apply(lambda x: len(str(x)))

# Count number of links in email content
df['link_count'] = df['Email_Content'].apply(lambda x: len(re.findall(r"http\\S+", str(x))))

# Extract sender domain from email address
df['sender_domain'] = df['Sender_Email'].apply(
    lambda x: str(x).split('@')[-1] if pd.notnull(x) else 'unknown'
)

# Encode sender domain as numeric feature
df['sender_domain_encoded'] = pd.factorize(df['sender_domain'])[0]

# Create binary feature: has_link (1 if link_count > 0, else 0)
df['has_link'] = df['link_count'].apply(lambda x: 1 if x > 0 else 0)

# Step 6: Define features for correlation analysis
features = ['subject_length', 'link_count', 'sender_domain_encoded']

# Step 7: Visualisation 1 – Histogram of Link Count
plt.figure(figsize=(8, 5))
sns.histplot(df['link_count'], bins=30, kde=True, color='skyblue')
plt.title("Histogram of Link Count")
plt.xlabel("Number of Links in Email")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Step 8: Visualisation 2 – Feature Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Step 9: Visualisation 3 – Bar Chart of Link Presence by Email Type
plt.figure(figsize=(6, 4))
sns.countplot(x='has_link', hue='Label', data=df, palette='Set2')
plt.title("Presence of Links in Phishing vs Legitimate Emails")
plt.xlabel("Has Link (0 = No, 1 = Yes)")
plt.ylabel("Email Count")
plt.legend(title='Email Type', labels=['Legitimate', 'Phishing'])
plt.tight_layout()
plt.show()

# Step 10: Visualisation 4 – Pie Chart of Overall Link Presence
# Count how many emails have links vs no links
link_counts = df['has_link'].value_counts()

# Map values to readable labels
label_map = {0: 'No Links', 1: 'Has Links'}
labels = [label_map.get(i, f'Unknown ({i})') for i in link_counts.index]

# Plot pie chart
plt.figure(figsize=(5, 5))
plt.pie(link_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgrey', 'lightcoral'])
plt.title("Proportion of Emails with Links")
plt.tight_layout()
plt.show()
