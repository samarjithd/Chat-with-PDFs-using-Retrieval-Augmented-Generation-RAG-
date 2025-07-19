import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extraction of text from the PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Conversion of text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=2000)
    return text_splitter.split_text(text)

# Converting the chunks into the vector
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Template for the output
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Don't provide incorrect answers.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    # ✅ Corrected model name
    model = ChatGoogleGenerativeAI(model="models/gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Complete generation of the output
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF using Gemini")

    if 'pdf_text_chunks' not in st.session_state:
        st.session_state.pdf_text_chunks = None

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    user_question = st.text_input("Ask a Question about the PDF")

    if user_question and st.session_state.pdf_text_chunks:
        response_text = user_input(user_question)
        st.session_state.conversation_history.append(f"**You:** {user_question}")
        st.session_state.conversation_history.append(f"**Gemini:** {response_text}")

    for message in st.session_state.conversation_history:
        st.markdown(message)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.pdf_text_chunks = text_chunks
                    st.session_state.conversation_history = []
                    st.success("PDF processed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
