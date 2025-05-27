'''Create vector store first.
'''
import streamlit as st
import os, glob, logging
import pickle, uuid
from pathlib import Path
import fitz
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from fred_rag_chatbot_config import fred_config
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_chatbot_retriever():
    if fred_config["prefer_pull_model"] == True:
            ollama.pull(fred_config["system1_model_name"])
            ollama.pull(fred_config["system2_model_name"])

    logging.info("System-1 and System-2 LLM models are pulled.")

    pdf_paths = sorted(list(Path(fred_config["doc_dir_path"]).glob("*.pdf")))
    if os.path.exists(fred_config["docstore_persist_directory"]):
        Path(fred_config["docstore_persist_directory"]).mkdir(parents=True, exist_ok=True)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=50,
    )
    logging.info("Parent splitter created.")
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=25,
    )
    logging.info("Child splitter created.")
    vectorstore = Chroma(
        collection_name=fred_config["vector_store_name"],
        embedding_function=OllamaEmbeddings(model=fred_config["embedding_model"]),
        persist_directory=fred_config["chromadb_persist_directory"],
    )
    logging.info("Vector store created")
    fs = LocalFileStore(fred_config["docstore_persist_directory"])
    store = create_kv_docstore(fs)
    logging.info("Docstore created")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 1},
    )
    logging.info("Parent Document Retriever created")
    docs = []

    for p in tqdm(pdf_paths):
        pdf_doc = fitz.open(str(p))
        pdf_str = ""
        for page in pdf_doc:
            pdf_str += page.get_text()

        # create langchian doc
        doc = Document(page_content=pdf_str)
        docs.append(doc)

    retriever.add_documents(docs)
    logging.info("Added documents.")

if __name__ == '__main__':
    get_chatbot_retriever()