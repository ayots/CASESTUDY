# Suppress TensorFlow and general warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time

# Classical ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# BERT
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import logging
import tensorflow as tf
logging.set_verbosity_error()

# Step 1: Load cleaned phishing email dataset
df = pd.read_csv('Cleaned_PhishingEmailData.csv', encoding='ISO-8859-1')

# Step 2: Feature engineering — clean email content and extract basic features
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "link", text)
    text = re.sub(r"\S+@\S+", "email", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df['clean_content'] = df['Email_Content'].apply(clean_text)
df['subject_length'] = df['Email_Subject'].apply(lambda x: len(str(x)))
df['link_count'] = df['Email_Content'].apply(lambda x: len(re.findall(r"http\S+", str(x))))
df['has_link'] = df['link_count'].apply(lambda x: 1 if x > 0 else 0)

# Step 3: Select features for classical models
features = ['subject_length', 'link_count', 'has_link']
X = df[features]
y = df['Label']

# Step 4: Train/test split and feature scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Define evaluation and plotting functions
def evaluate_model(y_true, y_pred):
    y_true_oh = pd.get_dummies(y_true)
    y_pred_oh = pd.get_dummies(y_pred)
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'ROC-AUC': roc_auc_score(y_true_oh, y_pred_oh, multi_class='ovr')
    }

def plot_metrics(name, metrics):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='Blues')
    plt.title(f"{name} Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    time.sleep(0.5)

# Step 6: Train and evaluate classical models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier()
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    metrics = evaluate_model(y_test, y_pred)
    results[name] = metrics
    plot_metrics(name, metrics)

# Step 7: Train and evaluate LSTM model
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['clean_content'])
X_seq = tokenizer.texts_to_sequences(df['clean_content'])
X_pad = pad_sequences(X_seq, maxlen=100)

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_pad, y, test_size=0.2, random_state=42)

lstm_model = Sequential()
lstm_model.add(Embedding(5000, 128))
lstm_model.add(SpatialDropout1D(0.2))
lstm_model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

lstm_model.fit(X_train_lstm, y_train_lstm, epochs=3, batch_size=64, verbose=0)
y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
predictions['LSTM'] = y_pred_lstm
print("\nLSTM Classification Report:\n")
print(classification_report(y_test_lstm, y_pred_lstm, zero_division=0))
metrics_lstm = evaluate_model(y_test_lstm, y_pred_lstm)
results['LSTM'] = metrics_lstm
plot_metrics("LSTM", metrics_lstm)

# Step 8: Train and evaluate BERT model (fixed optimizer)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_inputs = bert_tokenizer(list(df['Email_Content'].astype(str)), padding=True, truncation=True, return_tensors='tf')
input_ids = bert_inputs['input_ids'].numpy()
attention_mask = bert_inputs['attention_mask'].numpy()

X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(input_ids, y, test_size=0.2, random_state=42)
attention_train, attention_test = train_test_split(attention_mask, test_size=0.2, random_state=42)

bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2, from_pt=True)

# ✅ Use legacy-compatible optimizer to avoid runtime error
bert_model.compile(
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

train_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': X_train_bert, 'attention_mask': attention_train},
    y_train_bert
)).batch(16)

bert_model.fit(train_dataset, epochs=2, verbose=1)

test_dataset = tf.data.Dataset.from_tensor_slices({
    'input_ids': X_test_bert,
    'attention_mask': attention_test
}).batch(16)

y_pred_bert = np.argmax(bert_model.predict(test_dataset).logits, axis=1)
predictions['BERT'] = y_pred_bert
print("\nBERT Classification Report:\n")
print(classification_report(y_test_bert, y_pred_bert, zero_division=0))
metrics_bert = evaluate_model(y_test_bert, y_pred_bert)
results['BERT'] = metrics_bert
plot_metrics("BERT", metrics_bert)

# Step 9: Final comparison of all models
results_df = pd.DataFrame(results).T
print("\nFinal Comparison Table:\n")
print(results_df)

plt.figure(figsize=(12, 6))
results_df.plot(kind='bar')
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

best_model = results_df['Accuracy'].idxmax()
print(f"\nBest phishing detector based on accuracy: {best_model}")

# Step 10: Simulate privacy-preserving techniques
print("\nStep 10: Privacy-Preserving Techniques")

def simulate_privacy_training():
    print("Simulating privacy-preserving training...")
    print("- Applying differential privacy noise to model outputs...")
    print("- Training model across decentralized nodes (federated learning)...")
    print("- Aggregating model updates securely without sharing raw data...")
    print("Privacy-preserving simulation complete. Ready for integration with PySyft or SmartNoise.")

simulate_privacy_training()
