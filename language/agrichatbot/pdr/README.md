# Chatbot

A preliminary Retrieval-Augmented Generation (RAG) based Chatbot framework(using **ParentDocumentRetriever**) for agricultural business applications is developed using LangChain, LangGraph, Chroma, and Ollama. The PDF documents for agricultural business applications are used as *Fruit and Vegetable Quality* related PDFs from GS1 Standards, and Fresh Fruit and Vegetables Standards PDFs from United Nations Economic Commission for Europe (UNECE).

## Getting Started
### Installation
Please create conda environment and install required libraries:
```shell
# Install libraries
$ pip install -r requirements.txt
```

### Experiment
*At first*, please run vector store creation from PDF documents.

```shell
# Run vector store
$ python agrichatbot/pdr/fred_create_chatbot_vetorstore.py
```
*Afterthat*, please run Streamlit application for Chatbot:

```shell
# Run Streamlit app for QA
$ python agrichatbot/pdr/app.py
```

The configuration information is mentioned at *agrichatbot/pdr/fred_rag_chatbot_config.py* file. Please adapt the config file according to your choice.


### Approaches
Some approaches are used for preliminary framework:
* *Embedding Model*: Pull **nomic-embed-text** model from Ollama website.
* *LLM*: Pull **llama3.1** model for *System-1* and **deepseek-r1** for *System-2* from Ollama website.
* *Vector Store*: Chroma is used.
* *Document Store*: LocalFileStore and create_kv_docstore are used.
* *PDF Docs*: PyMuPDF is used to parse PDFs. Document parsing is a crucial tasks. There are several ways to improve it by analysing different refinement techniques.
* *Chunking*: RecursiveCharacterTextSplitter for splitting.
* *Retriever*: **ParentDocumentRetriever** is used. 
* Area of refinement approaches
    * PDF parsing, Retriever, Prompt, RAG chain etc. 