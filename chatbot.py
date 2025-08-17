import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings  # dense embeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.header("Naive RAG Chatbot")

with st.sidebar:
    st.title("Upload your documents")
    file = st.file_uploader("Upload your PDF", type="pdf")

# Initialize embeddings outside file loop so caching works
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # CPU-friendly dense retriever
    model_kwargs={"device": "cpu"}
)

index_path = "faiss_index"

if file is not None:
    # Extract text from PDF
    pdf_pages = PdfReader(file)
    text = ""
    for page in pdf_pages.pages:
        text += page.extract_text() or ""

    # Break text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Either load existing FAISS or build new one
    if os.path.exists(index_path):
        vector_store = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(index_path)

    # Get user query
    user_query = st.text_input("What is your query?")

    if user_query:
        # Fallback for self-identity questions
        fallback_triggers = ["who are you", "what can you do", "about you", "your purpose"]
        if any(trigger in user_query.lower() for trigger in fallback_triggers):
            st.subheader("Answer")
            st.write("I am a Naive RAG Chatbot 🤖. "
                     "I can read your uploaded PDF documents, create embeddings using HuggingFace, "
                     "store them in a FAISS dense vector index, and answer your questions "
                     "based on the retrieved chunks of text.")
        else:
            llm = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0.25,
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

            # Dense retrieval using FAISS
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_store.as_retriever(
                    search_type="similarity",  # dense similarity search
                    search_kwargs={"k": 3}
                ),
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": custom_prompt},
            )

            result = qa({"query": user_query})  # must be 'query'!

            # Show final answer
            st.subheader("Answer")
            st.write(result['result'])

            # Show sources
            st.subheader("Sources")
            for i, doc in enumerate(result["source_documents"], start=1):
                preview = doc.page_content[:300].replace("\n", " ")
                st.write(f"**Source {i}:** {preview}...")
else:
    st.write("Please upload a PDF document to start.")

