import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()

# Modern LangChain Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader

# Fix: Use langchain.chains or langchain_classic if you have it installed
from langchain.chains import RetrievalQA

# -------------------- SETTINGS --------------------
DATA_FILE = "SLC Full Course Tracker Sheet.xls"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="SLC Assistant", layout="centered")
st.title("South London College Chatbot")

# -------------------- DATA + VECTOR STORE --------------------
@st.cache_resource
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"File '{DATA_FILE}' not found. Please upload it to your folder.")
        return None

    try:
        df = pd.read_excel(DATA_FILE) 
        df = df.fillna("N/A")
        df.columns = df.columns.str.strip()

        # Build searchable text
        df["combined"] = df.apply(
            lambda r: (
                f"Course: {r.get('Course Name', 'N/A')} | "
                f"Category: {r.get('Main Categories', 'N/A')} | "
                f"Price: {r.get('Standard Sale Price', 'N/A')} | "
                f"URL: {r.get('Course URL', 'N/A')}"
            ),
            axis=1
        )

        loader = DataFrameLoader(df, page_content_column="combined")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return FAISS.from_documents(loader.load(), embeddings)

    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None

db = load_data()

# -------------------- CHAT LOGIC FUNCTION --------------------
def get_bot_response(query):
    if not db or not OPENAI_API_KEY:
        return "System Error: Missing Database or API Key."
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4})
    )
    
    response = qa.invoke(query)
    return response["result"]

# -------------------- MAKE.COM AUTOMATION --------------------
# This allows Make.com to get a response via the URL
if "question" in st.query_params:
    bot_answer = get_bot_response(st.query_params["question"])
    st.write(bot_answer)
    st.stop() # Stop here so it doesn't render the whole UI for Make.com

# -------------------- CHAT UI --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me about SLC courses")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_bot_response(user_input)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})