import streamlit as st
import os, glob, logging
from langchain_community.vectorstores import Chroma
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

# Load or create the vector database.
@st.cache_resource()
def load_vector_db():
    if fred_config["prefer_pull_model"] == True:
            ollama.pull(fred_config["embedding_model"])
    
    embedding = OllamaEmbeddings(model=fred_config["embedding_model"])
    # Load existing vector database
    if os.path.exists(fred_config["chromadb_persist_directory"]):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=fred_config["vector_store_name"],
            persist_directory=fred_config["chromadb_persist_directory"],
        )
        logging.info("Loaded existing vector database.")
        return vector_db

    else:
        return None

# Create a multi-query retriever.
def create_retriever(vector_db, model):
    query_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=fred_config["fred_contextual_promt_template"],
    )
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(search_type="mmr"), 
        llm=model, 
        prompt=query_prompt,
    )
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multiquery_retriever
    )

    logging.info("Retriever created.")
    return compression_retriever

# Create RAG chain
def create_chain(retriever, model):
    prompt=ChatPromptTemplate.from_template(fred_config["rag_prompt_template"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=model, retriever=retriever,
        return_source_documents = True,
        chain_type_kwargs = {
            "prompt": prompt
        }
    )
    logging.info("Chain created successfully with preserved syntax.")
    return qa_chain

def workflow_fred_mqr_qa():
    st.title("FRED Document Assistant")
    
    # User input
    user_input = st.text_input(label="Enter your question:", value="")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Initialize the language model
                model = ChatOllama(model=fred_config["model_name"])
                # Load the vector database
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load the vector database.")
                    return

                # Create the retriever
                retriever = create_retriever(vector_db, model)

                # Create the chain
                chain = create_chain(retriever, model)

                # Get the response
                response = chain.invoke(input=user_input)
                st.markdown("**FRED QA Assistant:**")
                st.write(response["result"])

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    else:
        st.info("Please enter a question to get started.")
    
if __name__ == '__main__':
    workflow_fred_mqr_qa()
