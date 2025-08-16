import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

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
        match=vector_store.similarity_search(user_query)
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.70,
            max_retries=2,
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_query)
        st.subheader("Answer")
        st.write(response)
    
else:
    st.write("Please upload a PDF document to start.")