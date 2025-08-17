import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings  # new recommended import
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.header("Naive RAG Chatbot")

with st.sidebar:
    st.title("Upload your documents")
    file = st.file_uploader("Upload your PDF", type="pdf")

if file is not None:
    # Extract text from PDF
    pdf_pages = PdfReader(file)
    text = ""
    for page in pdf_pages.pages:
        text += page.extract_text() or ""

    # Break text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings using the new HuggingFace class
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # CPU-friendly
        model_kwargs={"device": "cpu"}
    )

    # Create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user query
    user_query = st.text_input("What is your query?")

    if user_query:
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.0,
            max_retries=2,
        )

        # Prompt must use 'query' to match RetrievalQA default input key
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a document-based assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say exactly:
"I don’t know about this topic based on the uploaded document."

Context:
{context}

User Query:
{question}
Answer:
"""
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt},
        )

        result = qa({"query": user_query})  # must be 'query'!
        st.subheader("Answer")
        st.write(result['result'])
else:
    st.write("Please upload a PDF document to start.")
