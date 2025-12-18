# 📄 Chat with PDF using Retrieval-Augmented Generation (RAG)

This project implements a **PDF Question Answering system** using the **Retrieval-Augmented Generation (RAG)** approach.  
Users can upload PDF documents and ask questions to get **context-aware, document-grounded answers**.

The system combines **local embeddings**, a **FAISS vector database**, and a **large language model (LLM)** for accurate and reliable responses.

---

## 🚀 Features

- Upload and process multiple PDF files
- Automatic text chunking and vector embedding
- Semantic search using FAISS vector database
- Context-aware question answering (RAG)
- PDF summarization
- Streamlit-based interactive UI
- Uses free and open-source components where possible

---

## 🧠 Architecture Overview
PDF Files
->
Text Extraction (PyPDF2)
->
Text Chunking
->
Embeddings (HuggingFace MiniLM)
->
FAISS Vector Database (Local, Persistent)
->
Similarity Search
->
Context Injection
->
LLM Response (Groq - LLaMA 3.1)


---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Document Processing
- PyPDF2

### Chunking
- RecursiveCharacterTextSplitter (LangChain)

### Embeddings
- HuggingFace Sentence Transformers  
- Model: `sentence-transformers/all-MiniLM-L6-v2`  
- Runs locally (no embedding API calls)

### Vector Database
- FAISS (local, disk-persisted)

### LLM (Text Generation)
- Groq API  
- Model: `llama-3.1-8b-instant`

---

## 🔍 How RAG Works in This Project

1. PDF files are parsed and converted into text.
2. Text is split into overlapping chunks.
3. Each chunk is converted into a vector embedding using a local HuggingFace model.
4. Embeddings are stored in a FAISS vector database on disk.
5. When a user asks a question:
   - The query is embedded.
   - FAISS retrieves the most relevant chunks using similarity search.
   - Retrieved chunks are passed as context to the LLM.
6. The LLM generates an answer strictly based on the retrieved context.

This approach reduces hallucination and ensures document-grounded answers.

---

## 🗂️ Vector Storage Details

- FAISS is used as the vector database.
- Embeddings are stored locally and persist across sessions.
- The FAISS index is rebuilt when new PDFs are uploaded.
- No external vector database services are used.

---

## 📝 PDF Summarization

The application also supports PDF summarization.
This feature directly sends the extracted text to the LLM and does not use retrieval.

---

## ▶️ How to Run the Project
1. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate
```
2.Create a .env file in the project root and add your Groq API key:
```bash
GROQ_API_KEY=your_groq_api_key_here
```
3.Install the required dependencies & Run
```bash
pip install -r requirements.txt
streamlit run chat.py
```
---









