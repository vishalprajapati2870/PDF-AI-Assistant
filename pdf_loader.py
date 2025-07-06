from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_DIR = "chroma_store"

def load_and_store_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    # Split into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Embed using HuggingFace (can use all-MiniLM for lightweight)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Store in ChromaDB
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()

    return vectordb

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectordb
