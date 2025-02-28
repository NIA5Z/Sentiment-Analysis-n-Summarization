import streamlit as st
from gpt4all import GPT4All

st.set_page_config(page_title="Chat", page_icon="💡", layout="wide", initial_sidebar_state="collapsed", menu_items={'Get Help': None,'Report a bug': None,'About': None})

GPT = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

st.title("🤖 Chat with AI")

extracted_text = st.session_state.get("extracted_text", "")
sentiment = st.session_state.get("sentiment", "No Sentiment available.")
summary = st.session_state.get("summary", "No summary available.")

if "history" not in st.session_state:
    st.session_state.history = []

if "response" not in st.session_state:
    st.session_state.response = ""

st.subheader("📊 Sentiment Analysis Result")
st.markdown(f"**Sentiment:** {sentiment}")

st.subheader("📜 Summary")
st.write(summary)

user_input = st.text_input("Ask a question based on the summary:")

col1, col2 = st.columns([1,20])

with col1:
    if st.button("Send"):
        if user_input.strip():
            chat_prompt = f"""
            Sentiment: {sentiment}
            Context: {summary}
            Context History: {st.session_state.history}
            User Query: {user_input}
            """
            
            with GPT.chat_session():
                response = GPT.generate(chat_prompt, max_tokens=1024)

            st.session_state.history.append({'User': user_input, 'AI': response})
            st.session_state.response = response
        else:
            st.warning("⚠️ Please enter a question to continue chatting.")

with col2:
    if st.button("Return"):
        st.switch_page('./Main.py')

st.subheader("🤖 AI Response")
st.write(st.session_state.response)
st.subheader("📜 Logs")
st.write(st.session_state.history)