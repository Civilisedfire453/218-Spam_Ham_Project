import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (Assuming it's in the same folder as this script)
data = pd.read_csv('spam_ham_dataset.csv')

# Check the first few rows to see what the data looks like
print(data.head())

X = data['text']  # the email content
y = data['label']  # the labels (spam or ham)

# Spliting the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert the email texts into numbers using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')  # Ignore common words like 'the', 'is', etc.
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Naive Bayes model (a simple machine learning model)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test_tfidf)

# Check how well the model did
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Show a confusion matrix to see where the model got confused
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.show()

# Save the trained model to a file so we can use it later without retraining
#joblib.dump(nb_classifier, 'spam_model.pkl')
# Save the vectorizer along with the model
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(nb_classifier, 'spam_model.pkl')

# You can load the saved model like this if you want to use it later
# loaded_model = joblib.load('spam_model.pkl')
# y_pred_loaded = loaded_model.predict(X_test_tfidf)
# print("Accuracy with loaded model:", accuracy_score(y_test, y_pred_loaded))
