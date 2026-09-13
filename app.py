"""
Optional web UI for the medical RAG system, built with Streamlit.

Run with:
    streamlit run app.py

This reuses every function from 03_rag_chat.py -- the web UI is just a
thin presentation layer on top of the same retrieval + LLM logic.
"""

import streamlit as st

from importlib import import_module

rag = import_module("03_rag_chat")  # loads 03_rag_chat.py from the same folder

st.set_page_config(page_title="Medical Guidelines RAG", page_icon="🩺")
st.title("Medical guidelines Q&A (RAG demo)")
st.caption(
    "Educational demo only -- not medical advice. Answers are grounded in "
    "a small sample of public clinical guidelines (WHO, CDC, WikiDoc)."
)


@st.cache_resource
def load_retriever():
    return rag.get_retriever()


retriever = load_retriever()

question = st.text_input("Ask a question about the indexed guidelines:")

if question:
    with st.spinner("Retrieving relevant guideline passages..."):
        docs = retriever.invoke(question)

    with st.spinner("Generating a grounded answer..."):
        prompt = rag.build_prompt(question, docs)
        answer = rag.call_llm(prompt)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources retrieved")
    for i, doc in enumerate(docs, start=1):
        with st.expander(f"[{i}] {doc.metadata.get('title', 'unknown')}"):
            st.write(doc.page_content)
