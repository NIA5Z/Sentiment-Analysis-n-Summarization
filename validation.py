import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D,Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
import pickle
import numpy as np

nltk.download("punkt_tab")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

df = pd.read_csv("./Dataset/Tweets.csv", encoding="ISO-8859-1")
df = df.drop(df.columns[[0,2]], axis=1)
df.columns = ["tweet_text","sentiment"]
df["sentiment"] = df["sentiment"].replace(4, 1)
df = df.dropna()

sentiment_mapping = {"neutral": 0, "negative": 1, "positive": 2}
df["sentiment"] = df["sentiment"].map(sentiment_mapping)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df["tweet_text"] = df["tweet_text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df["tweet_text"], df["sentiment"], test_size=0.2, random_state=42)

# Load tokenizer
with open("./Model/word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>", lower=True, filters='')
tokenizer.word_index = word_index

# Load test data
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200, padding='post')

# Load model
model = load_model("./Model/twitter_Sentimental1M.keras")

# Make predictions
y_pred = model.predict(X_test_seq)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy:.4f}")


# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))
