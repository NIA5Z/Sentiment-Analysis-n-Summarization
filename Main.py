import re
import nltk
import pickle
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gpt4all import GPT4All

# Download NLTK data if not available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Model constants
MAX_NUM_WORDS = 20000
MAX_SEQ_LENGTH = 200

# Load tokenizer and sentiment model
with open("./Model/word_index.pkl", "rb") as f:
    word_index = pickle.load(f)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.word_index = word_index

model = load_model("./Model/twitter_Sentimental1M.keras")
GPT = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE) 
    text = re.sub(r'\@\w+|\#', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    words = word_tokenize(text) 
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Sentiment prediction function
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding='post')
    prediction = model.predict(padded_seq)
    sentiment = np.argmax(prediction, axis=1)[0]
    return (
        "Positive" if sentiment == 2 
        else "Negative" if sentiment == 1 
        else "Neutral"
    )

# Web scraping function
def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None, "Failed to fetch the webpage. Please check the URL."
        
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract text from <p> tags
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text.strip() if text else None, "No readable content found on the webpage."
    
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching URL: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis & Summarization", page_icon="💡", layout="wide", initial_sidebar_state="collapsed", menu_items={'Get Help': None,'Report a bug': None,'About': None})

st.title("📊 Sentiment Analysis & Summarization")
st.markdown("Enter your post text **OR** provide a webpage URL to analyze.")
# Option selection: Manual Input vs Web Scraping
option = st.radio("Choose Input Method:", ("Enter Text", "Scrape from URL"))

if option == "Enter Text":
    text_input = st.text_area("Enter post text:", height=150)
else:
    url_input = st.text_input("Enter a webpage URL:")

if st.button("Analyze 🧐"):
    extracted_text = None
    user = None

    if option == "Enter Text":
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            extracted_text = text_input

    else:  # Web Scraping Mode
        if not url_input.strip():
            st.warning("⚠️ Please enter a valid URL.")
        else:
            extracted_text, error_message = scrape_text_from_url(url_input)
            if not extracted_text:
                st.error(f"❌ {error_message}")
    
    # If valid text is available, proceed with analysis
    if extracted_text:
        sentiment = predict_sentiment(extracted_text)
        color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "white"
        
        formatted_prompt = f"""Use the context to answer the given instruction.
        
        Sentiment: {sentiment}
        Context: {extracted_text}
        User: {user}
        """
        layout="wide", 
        with GPT.chat_session(system_prompt="You are to summarize the given post concisely while preserving key information and main points"):
            summary = GPT.generate(formatted_prompt, max_tokens=1024)
        
        st.subheader("🔍 Sentiment Analysis Result")
        st.markdown(f"<h3 style='color:{color};'>{sentiment}</h3>", unsafe_allow_html=True)
        
        st.subheader("📜 Summarized Text")
        st.write(summary)
        
        st.session_state["extracted_text"] = extracted_text
        st.session_state["sentiment"] = sentiment
        st.session_state["summary"] = summary

        st.switch_page('./pages/Chat.py')