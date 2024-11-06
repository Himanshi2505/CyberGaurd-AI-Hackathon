import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Set the path for NLTK to use Kaggle's data directory
nltk.data.path.append("/usr/share/nltk_data")

# Load datasets (adjust file paths as per your Kaggle input folder)
train_data = pd.read_csv("/kaggle/input/traindata/train.csv")
test_data = pd.read_csv("/kaggle/input/test-data/test.csv")

# Preprocessing function
def preprocess_text(text):
    if pd.isna(text):  # Check for NaN
        return ""      # Return empty string for NaN values
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Apply preprocessing
train_data['processed_text'] = train_data['crimeaditionalinfo'].apply(preprocess_text)
test_data['processed_text'] = test_data['crimeaditionalinfo'].apply(preprocess_text)

# Drop rows with missing values in 'category' or 'sub_category' columns in the training set
train_data.dropna(subset=['category', 'sub_category'], inplace=True)

# Drop rows with missing values in 'category' or 'sub_category' columns in the test set
test_data.dropna(subset=['category', 'sub_category'], inplace=True)

# Prepare features and labels
X_train = train_data['processed_text']
y_train_category = train_data['category']
y_train_sub_category = train_data['sub_category']
X_test = test_data['processed_text']
y_test_category = test_data['category']
y_test_sub_category = test_data['sub_category']

# Vectorize text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model for 'category'
model_category = MultinomialNB()
model_category.fit(X_train_vec, y_train_category)

# Train Naive Bayes model for 'sub_category'
model_sub_category = MultinomialNB()
model_sub_category.fit(X_train_vec, y_train_sub_category)

# Predict and evaluate for 'category'
y_pred_category = model_category.predict(X_test_vec)
accuracy_category = accuracy_score(y_test_category, y_pred_category)
precision_category = precision_score(y_test_category, y_pred_category, average='weighted')
recall_category = recall_score(y_test_category, y_pred_category, average='weighted')
f1_category = f1_score(y_test_category, y_pred_category, average='weighted')
report_category = classification_report(y_test_category, y_pred_category)

# Predict and evaluate for 'sub_category'
y_pred_sub_category = model_sub_category.predict(X_test_vec)
accuracy_sub_category = accuracy_score(y_test_sub_category, y_pred_sub_category)
precision_sub_category = precision_score(y_test_sub_category, y_pred_sub_category, average='weighted')
recall_sub_category = recall_score(y_test_sub_category, y_pred_sub_category, average='weighted')
f1_sub_category = f1_score(y_test_sub_category, y_pred_sub_category, average='weighted')
report_sub_category = classification_report(y_test_sub_category, y_pred_sub_category)

# Save results to file
with open("/kaggle/working/output.txt", "w") as f:
    f.write("Category Classification Metrics:\n")
    f.write(f"Accuracy: {accuracy_category}\n")
    f.write(f"Precision: {precision_category}\n")
    f.write(f"Recall: {recall_category}\n")
    f.write(f"F1 Score: {f1_category}\n")
    f.write("\nClassification Report for Category:\n")
    f.write(report_category)
    
    f.write("\n\nSub-Category Classification Metrics:\n")
    f.write(f"Accuracy: {accuracy_sub_category}\n")
    f.write(f"Precision: {precision_sub_category}\n")
    f.write(f"Recall: {recall_sub_category}\n")
    f.write(f"F1 Score: {f1_sub_category}\n")
    f.write("\nClassification Report for Sub-Category:\n")
    f.write(report_sub_category)

print("Model evaluation metrics have been saved to 'output.txt'.")
