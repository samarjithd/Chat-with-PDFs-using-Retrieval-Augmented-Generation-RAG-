import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

import requests  # for Groq API calls

# ----------------- CONFIG -----------------

load_dotenv()  # loads GROQ_API_KEY, etc.


def get_embeddings():
    # Local, free, small + good
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def call_groq(prompt: str) -> str:
    """
    Call Groq's chat completion API with a simple user prompt.
    Uses LLaMA-3.1-8B-Instant (good balance of speed + quality).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")

    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    res = requests.post(url, json=payload, headers=headers)
    res.raise_for_status()
    data = res.json()

    return data["choices"][0]["message"]["content"]


# ----------------- STREAMLIT APP -----------------

def handle_question_change():
    question = st.session_state.user_question_input
    if question:
        user_input(question)          # generate answer
        st.session_state.user_question_input = ""  # clear input


def main():
    st.set_page_config(page_title="Chat PDF", page_icon="💁")
    st.header("Chat with PDF")

    # text_input with callback
    st.text_input(
        "Ask a Question from the PDF Files",
        key="user_question_input",
        on_change=handle_question_change,
    )

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
            type="pdf"
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    update_vector_store(text_chunks)
                    st.session_state["pdf_text"] = raw_text
                    st.success("Done")
            else:
                st.warning("Please upload at least one PDF file.")

    if "pdf_text" in st.session_state and st.button("Summarize PDF"):
        summary = summarize_text(st.session_state["pdf_text"])
        st.subheader("PDF Summary")
        st.write(summary)



# ----------------- PDF HELPERS -----------------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


# ----------------- VECTOR STORE (FAISS) -----------------

def update_vector_store(text_chunks):
    import shutil

    embeddings = get_embeddings()

    # Always rebuild FAISS from scratch to avoid dim mismatch weirdness
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# ----------------- LLM: SUMMARIZATION (GROQ) -----------------

def summarize_text(text: str) -> str:
    """
    Summarize the whole PDF text using Groq (LLaMA-3.1-8B).
    """
    prompt = (
        "You are a helpful assistant. Summarize the following PDF content "
        "clearly and concisely, keeping all key points and structure. "
        "Avoid unnecessary fluff. If the text is long, focus on the main sections "
        "and key ideas.\n\n"
        f"{text}"
    )

    return call_groq(prompt)


# ----------------- USER Q&A FLOW (GROQ) -----------------

def user_input(user_question):
    embeddings = get_embeddings()

    if "faiss_index" not in os.listdir():
        st.error("No FAISS index found. Please upload and process PDFs first.")
        return

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question, k=4)

    if not docs:
        st.write("Reply: answer is not available in the context")
        return

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a question-answering assistant.

Use ONLY the information in the context below to answer the question.
If the answer is not present in the context, reply exactly with:
"answer is not available in the context".

Be detailed but precise.

Context:
{context}

Question:
{user_question}

Answer:
"""

    answer = call_groq(prompt)

    st.write("Reply:", answer)


# ----------------- ENTRYPOINT -----------------

if __name__ == "__main__":
    main()

