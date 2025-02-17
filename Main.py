import re
import nltk
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gpt4all import GPT4All

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

MAX_NUM_WORDS = 20000
MAX_SEQ_LENGTH = 200

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    words = word_tokenize(text)  # Tokenize text
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(words)

with open("./Model/word_index.pkl", "rb") as f:
    word_index = pickle.load(f)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.word_index = word_index

model = load_model("./Model/twitter_Sentimental1M.keras")
GPT = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding='post')
    print(padded_seq)
    prediction = model.predict(padded_seq)
    sentiment = np.argmax(prediction, axis=1)[0]  # Get the predicted label
    return (
    "Positive" if sentiment == 2 
    else "Negative" if sentiment == 1 
    else "Neutral"
)

prompt_template ="""Use the context to answer the given instruction.

"Sentiment": {sentiment}
Context: {text_input}
"""

st.set_page_config(page_title="Sentiment Analysis & Summarization", page_icon="💡", layout="centered")

st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5;}
    h1 {color: #333366; text-align: center;}
    .stTextArea textarea {border-radius: 10px;}
    .stButton>button {background-color: #4CAF50; color: white; padding: 10px; border-radius: 10px;}
    .stButton>button:hover {background-color: #45a049;}
    .stMarkdown {text-align: center; font-size: 18px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Sentiment Analysis & Summarization")
st.markdown("Enter your post text below to analyze its sentiment and generate a summary.")

text_input = st.text_area("Enter post text:", height=150)

if st.button("Analyze 🧐"):
    if text_input:
        sentiment = predict_sentiment(text_input)
        color = (
            "green" if sentiment == "Positive" 
            else "red" if sentiment == "Negative" 
            else "white"
        )
        formatted_prompt = prompt_template.format(sentiment=sentiment, text_input=text_input)

        with GPT.chat_session(system_prompt="You are to summarize the given post concisely while preserving key information and main points"):
            summary = GPT.generate(formatted_prompt, max_tokens=1024)
        
        st.subheader("🔍 Sentiment Analysis Result")
        st.markdown(f"<h3 style='color:{color};'>{sentiment}</h3>", unsafe_allow_html=True)
        
        st.subheader("📜 Summarized Text")
        st.write(summary)
    else:
        st.warning("⚠️ Please enter some text to analyze.")