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
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt_tab")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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
    words = word_tokenize(text)  # Tokenize text
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(words)

df["tweet_text"] = df["tweet_text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df["tweet_text"], df["sentiment"], test_size=0.2, random_state=42)

MAX_NUM_WORDS = 20000
MAX_SEQ_LENGTH = 200

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>", lower=True, filters='')
tokenizer.fit_on_texts(X_train)

with open("word_index.pkl", "wb") as f:
    pickle.dump(tokenizer.word_index, f)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_SEQ_LENGTH, padding='post')
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_SEQ_LENGTH, padding='post')

class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

model = Sequential([
    Embedding(input_dim=MAX_NUM_WORDS, output_dim=256, input_length=MAX_SEQ_LENGTH),
    SpatialDropout1D(0.3),
    
    Conv1D(128, 5, activation='relu', padding='same'),
    Conv1D(64, 3, activation='relu', padding='same'),
    
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64, return_sequences=False)), 
    
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

history = model.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test), epochs=20, batch_size=256, verbose=1,
                    class_weight=class_weight_dict, callbacks=[early_stopping, checkpoint, reduce_lr])

model.save("twitter_Sentimental1M.keras")