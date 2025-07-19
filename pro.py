import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit app setup
def main():
    st.set_page_config(page_title="Chat PDF", page_icon="💁")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

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
                    st.session_state["pdf_text"] = raw_text  # Store text for summarization
                    st.success("Done")
            else:
                st.warning("Please upload at least one PDF file.")

    if "pdf_text" in st.session_state and st.button("Summarize PDF"):
        summary = summarize_text(st.session_state["pdf_text"])
        st.subheader("PDF Summary")
        st.write(summary)

# Extraction of text from the PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Conversion of text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Updating vector store with new chunks from PDFs
def update_vector_store(text_chunks):
    # Check if FAISS index exists already
    if "faiss_index" in os.listdir():
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts([], embedding=embeddings)  # Start with an empty vector store
    
    # Add new chunks to the vector store
    vector_store.add_texts(text_chunks, embedding=embeddings)
    
    # Save the updated vector store
    vector_store.save_local("faiss_index")

# Summarization function
def summarize_text(text):
    prompt = "Summarize the following text concisely while retaining key points:\n" + text
    model = genai.GenerativeModel("models/gemini-1.5-flash")  # Correct API call for latest model
    response = model.generate_content(prompt)
    return response.text

# Template for the output
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n
#     Answer:
#     """
#     model = genai.GenerativeModel("gemini-pro")  # Updated model call
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain


from langchain_google_genai import ChatGoogleGenerativeAI

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(  # ✅ Proper wrapper for LangChain
        model="gemini-1.5-flash",  # or gemini-pro if your project supports
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Complete generation of the output
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load existing vector store
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Search for relevant documents based on the user question
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])

if __name__ == "__main__":
    main()
