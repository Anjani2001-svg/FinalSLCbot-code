import os
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader

DATA_FILE = "SLC Full Course Tracker Sheet.xls"

def build_db():
    """Build and return the FAISS vector DB (same as Streamlit, but without Streamlit)."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"'{DATA_FILE}' not found. Put it in the api/ folder.")

    df = pd.read_excel(DATA_FILE)
    df = df.fillna("N/A")
    df.columns = df.columns.str.strip()

    df["combined"] = df.apply(
        lambda r: (
            f"Course: {r.get('Course Name', 'N/A')} | "
            f"Category: {r.get('Main Categories', 'N/A')} | "
            f"Price: {r.get('Standard Sale Price', 'N/A')} | "
            f"URL: {r.get('Course URL', 'N/A')}"
        ),
        axis=1,
    )

    loader = DataFrameLoader(df, page_content_column="combined")

    # Reads OPENAI_API_KEY from env automatically
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(loader.load(), embeddings)
    return db
