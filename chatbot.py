import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()  # load variables from .env
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") 

st.header("Naive RAG Chatbot")

with st.sidebar:
    st.title("upload your documents")
    file=st.file_uploader("Upload your documents", type= "pdf")

#extract text from pdf
if file is not None:
    pdf_pages = PdfReader(file)
    text = ""
    for page in pdf_pages.pages:
        text += page.extract_text()
        #st.write(text)


# break the text into chunks
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len

    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

#generate embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )

# create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

# get user query
    user_query = st.text_input("What is your query?")

    if user_query:
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.45,
            max_retries=2,
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # number of documents to retrieve   
            ),
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": """You are a document-based assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say exactly:
"I don’t know about this topic based on the uploaded document."

Context:
{context}

Question: {question}
Answer:"""
            },
        )
       
        result = qa({"query": user_query})
        st.subheader("Answer")
        st.write(result['result'])
    
else:
    st.write("Please upload a PDF document to start.")