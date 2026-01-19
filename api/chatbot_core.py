import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_db = None  # will be injected on startup


def set_db(db):
    global _db
    _db = db


def generate_reply(user_text: str) -> str:
    if not _db or not OPENAI_API_KEY:
        return "System Error: Missing Database or API Key."

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    system_prompt = (
        "You are an assistant for South London College. "
        "Use the retrieved context to answer the user's question. "
        "If you don't know the answer, say you don't know. Keep it professional.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(_db.as_retriever(), question_answer_chain)

    result = rag_chain.invoke({"input": user_text})
    return result.get("answer", "I couldn't generate an answer.")


