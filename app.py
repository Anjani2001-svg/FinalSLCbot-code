import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()

# --- MODERN LANGCHAIN IMPORTS (v1.0+) ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader

# Updated imports for LangChain 2026 / v1.x
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Only use this if you explicitly need the old RetrievalQA
# from langchain_classic.chains import RetrievalQA 

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
        # Note: For .xls files in Python 3.11, you may need: pip install xlrd
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

    # 1. Create a System Prompt (Modern LCEL Format)
    system_prompt = (
        "You are an assistant for South London College. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer, say you don't know. Keep it professional."
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 2. Build the modern chain (Recommended over RetrievalQA)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(db.as_retriever(), question_answer_chain)
    
    # 3. Get the response
    result = rag_chain.invoke({"input": query})
    return result["answer"]

# -------------------- MAKE.COM AUTOMATION --------------------
# Updated for Streamlit 1.30+ query parameter handling
if st.query_params.get("question"):
    bot_answer = get_bot_response(st.query_params["question"])
    st.write(bot_answer)
    st.stop() 

# -------------------- CHAT UI --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
user_input = st.chat_input("Ask me about SLC courses")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_bot_response(user_input)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})