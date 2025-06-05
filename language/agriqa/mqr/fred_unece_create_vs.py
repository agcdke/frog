'''
Create vector embedding on UNECE "Fresh Fruit and Vegetables Standards" PDF documents.
'''
import os, glob, logging
import json, uuid, pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import ollama
import re, gc
from fred_rag_mqr_config import fred_config
from pdfplumber.utils import get_bbox_overlap, obj_to_bbox

# Configure logging
logging.basicConfig(level=logging.INFO)
    
# Remove unicode    
def remove_unicode(text):
    '''
    Remove unocode characters.
    text: str
    '''
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Convert document table to markdown text format, if document table structure (identified by pdfplumber) 
# is not being converted to pandas df_json (with orient="index") for indexing purpose.
def convert_doctable_to_mdtext(page):
    '''
    Convert document table to markdown text.
    page: pdfplumber page object
    '''
    for table in page.find_tables():
        filtered_page = page
        filtered_page = filtered_page.filter(lambda obj: 
                    get_bbox_overlap(obj_to_bbox(obj), table.bbox) is None
                )
        df = pd.DataFrame(table.extract())
        df.columns = df.iloc[0]
        markdown_df = df.drop(0).to_markdown(index=False)
        return markdown_df
    
# Extract elements from PDF
def extract_pdf_text_tables(dir_path, filename):
    '''
    Extract text and tables from a PDF file.
    path: directory path of PDF files
    fname: PDF filename
    '''
    text_list = []
    table_list = []
    pdf_file = os.path.join(dir_path, filename)
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            # Text
            if page_text:
                text_list.append(page_text)
                
            # Table
            table = page.extract_tables()
            if table:
                # Assume first row is header for UNECE STANDARD documents
                table_df = pd.DataFrame(table[0][1:], columns=table[0][0])
                table_df = table_df.reset_index(drop=True)
                try:
                    table_doc_list = [json.loads(remove_unicode(row.to_json(force_ascii=False, orient="index")))
                            for _, row in table_df.iterrows()]
                    table_list.append(table_doc_list)
                except ValueError as te:
                    print(f"Convert doc-table to markdown-text format as TypeError occrured: {te}")
                    markdown_df = convert_doctable_to_mdtext(page=page)
                    text_list.append(markdown_df)

    if text_list:
        joined_text = " ".join(text_list)
    else:
        joined_text = None
    
    if not table_list: 
        table_list = list()

    return joined_text, table_list

# Split document text and tables into smaller chunks
def split_text(docs):
    '''
    Split documents into chunks.
    docs: Document text
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=fred_config["chunk_size"], 
                                                   chunk_overlap=fred_config["chunk_overlap"])
    text_chunks = text_splitter.split_text(docs)
    logging.info("Split texts from RecursiveCharacterTextSplitter.")
    return text_chunks

def split_table_to_json(table_docs):
    '''
    Split document tables and convert into JSON format.
    table_docs: Document tables
    '''
    table_splitter = RecursiveJsonSplitter(max_chunk_size=fred_config["chunk_size"])
    table_chunks = table_splitter.split_json(json_data=table_docs, convert_lists=True)
    logging.info("Split tables from RecursiveJsonSplitter.")
    return table_chunks

# Load single PDF document.
def get_text_table_from_pdf_files(dir_path, filename):
    '''
    Extract text and tables from documents, and create LangChain Document object with id_key.
    dir_path: directory of PDF files
    filename: PDF filename
    '''
    id_key = "doc_id"
    if os.path.exists(dir_path):
        pdf_file = os.path.join(dir_path, filename)
        if os.path.exists(pdf_file):
            # Get text, tables
            text, tables = extract_pdf_text_tables(dir_path, filename)
            # Get doc chunks and table chunks for vectorstore
            if text:
                doc_chunks = split_text(text)
                logging.info("Split text into text chunks.")
                text_doc_ids = [str(uuid.uuid4()) for _ in doc_chunks]
                doc_text_chunks = [
                    Document(page_content=string_val, metadata={id_key: text_doc_ids[i]})
                    for i, string_val in enumerate(doc_chunks)
                ]
                logging.info("Create LangChain Document for Text.")
            else:
                doc_text_chunks=None
                logging.info("No text split.")

            if tables:
                table_chunks = split_table_to_json(tables)
                logging.info("Split table into table chunks.")
                # Convert a list of collections of JSON into a list of JSON
                table_json_chunks = [v for element in table_chunks
                                    for _,v in element.items()]
                logging.info("Create table JSON chunks.")
                
                table_doc_ids = [str(uuid.uuid4()) for _ in table_json_chunks]
                doc_table_chunks = [
                    Document(page_content=str(json_val), metadata={id_key: table_doc_ids[i]})
                    for i, json_val in enumerate(table_json_chunks)
                ]
                logging.info("Create LangChain Document for Table.")

            else:
                doc_table_chunks=None
                logging.info("No table JSON split.")
            return doc_text_chunks, doc_table_chunks
    else:
        logging.error(f"PDF documents not found at path: {dir_path}")
        st.error("PDF documents not found.")
        return None

def get_vector_db(embedding, chunks):
    '''
    Add documents to vector store.
    embedding: model embedding
    chunks: document (text or/and table) chunks
    '''
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

# Load or create the vector database.
def load_vector_db():
    '''
    Create vector database or load existing vector database to add documents.
    '''
    if fred_config["prefer_pull_model"] == True:
            ollama.pull(fred_config["embedding_model"])

    embedding = OllamaEmbeddings(model=fred_config["embedding_model"])

    # Load PDF documents.
    if os.path.exists(fred_config["doc_dir_path"]):
        pdf_files = [os.path.basename(filename) for filename in glob.glob(os.path.join(
                            fred_config["doc_dir_path"], '*.pdf'))]
        
        for filename in pdf_files:
            logging.info(f"Selected file: {filename}")
            # Get doc chunks, text, and tables after loading and processing the PDF documents
            doc_chunks, table_chunks = get_text_table_from_pdf_files(
                                                    fred_config["doc_dir_path"], 
                                                    filename)
            logging.info("Get text chunks, and table chunks")

            if doc_chunks:
                vector_db = get_vector_db(embedding, doc_chunks)
                logging.info("Store text chunks.")

            if table_chunks:
                vector_db = get_vector_db(embedding, table_chunks)
                logging.info("Store table chunks.")

            else:
                logging.info("Can not store text  and table chunks.")
        
# Worlflow: Create Vector Embedding
def workflow_create_vector_embedding():
    '''
    Workflow to create vector embedding.
    '''
    try:
        # Load the vector database
        load_vector_db()
        logging.info("Store relevant chunks to vector database.")
    except Exception as e:
        logging.info("An error occurred: ", str(e)) 

    _ = gc.collect()
    
    
if __name__ == '__main__':
    workflow_create_vector_embedding()
    