'''
Create vector store first.
'''
import streamlit as st
import os, glob, logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
import ollama
from langchain.chains import RetrievalQA
from fred_mqr_rag_config import fred_config

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load PDF documents.
def load_pdf_files(dir_path):
    if os.path.exists(dir_path):
        pdf_files = [filename for filename in glob.glob(os.path.join(dir_path, '*.pdf'))]
        docs = []
        for file in pdf_files:
            loader = UnstructuredPDFLoader(file_path=file, mode="single")
        docs.extend(loader.load())
        logging.info("PDF documents loaded successfully.")
        return docs
    else:
        logging.error(f"PDF documents not found at path: {dir_path}")
        st.error("PDF documents not found.")
        return None

# Split documents into smaller chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    logging.info("Documents split into chunks.")
    return chunks

# Load or create the vector database.
@st.cache_resource()
def load_vector_db():
    if fred_config["prefer_pull_model"] == True:
            ollama.pull(fred_config["embedding_model"])

    embedding = OllamaEmbeddings(model=fred_config["embedding_model"])
    # Load and process the PDF document
    data = load_pdf_files(fred_config["doc_dir_path"])
    if data is None:
        return None

    # Split documents into chunks
    chunks = split_documents(data)
    
    # Load existing vector database
    if os.path.exists(fred_config["chromadb_persist_directory"]):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=fred_config["vector_store_name"],
            persist_directory=fred_config["chromadb_persist_directory"],
        )
        vector_db.add_documents(documents=chunks)
        logging.info("Loaded existing vector database.")

    else:
        # Create new vector database and add chunks
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=fred_config["vector_store_name"],
            persist_directory=fred_config["chromadb_persist_directory"],
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db

def workflow_create_multi_query_retriever_embedding():
    try:
        # Load the vector database
        vector_db = load_vector_db()
        if vector_db is None:
            logging.info("Failed to load or create the vector database.")
            print("Failed to load or create the vector database.")

    except Exception as e:
        logging.info("An error occurred: ", str(e)) 
    
if __name__ == '__main__':
    workflow_create_multi_query_retriever_embedding()
