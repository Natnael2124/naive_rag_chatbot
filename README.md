# 📄 Naive RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, and **FAISS**.  
It allows you to upload a PDF document, process it into embeddings, and then ask questions about its content.

---

## 🚀 Features
- Upload your own **PDF documents**.
- Extracts and chunks text using **LangChain’s text splitter**.
- Generates embeddings with **HuggingFace Sentence Transformers**.
- Stores embeddings in a **FAISS vector database**.
- Answers user questions using **retrieval-based QA**.

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/naive-rag-chatbot.git
cd naive-rag-chatbot
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

Example Workflow

Upload a PDF from the sidebar.

The app extracts and chunks the text.

Ask a question about the document in the chat input.

The chatbot retrieves relevant chunks and answers.

📂 Project Structure
.
├── app.py                # Streamlit app
├── requirements.txt      # Dependencies
├── .env                  # (Optional) Environment variables
└── README.md             # Project documentation

📖 Example Questions

"Summarize this document in 3 sentences."

"What does it say about AI agents?"

"List the key challenges mentioned."

⚡ Tech Stack

Streamlit – UI framework

LangChain – RAG pipeline

FAISS – Vector database

Sentence Transformers – Embeddings