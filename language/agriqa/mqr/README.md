# Question-Answering

A preliminary Retrieval-Augmented Generation (RAG) based Question-Answering(QA) framework(using **MultiQueryRetriever**) for agricultural business applications is developed using LangChain, Chroma, and Ollama. The PDF documents for agricultural business applications are used as *Fruit and Vegetable Quality* related PDFs from *Fresh Fruit and Vegetables Standards* PDFs from [UNECE](https://unece.org/trade/wp7/FFV-Standards).

## Getting Started
### Installation
Please create conda environment and install required libraries:
```shell
# Install libraries
$ pip install -r requirements.txt
```

### Experiment
*At first*, please run vector store creation from PDF documents of UNECE dataset. The document table structure is adapted to store tabular data, as well as, textual data into Chroma database.

```shell
# Run vector store
$ python agriqa/mqr/fred_unece_create_vs.py
```
*Afterthat*, please run Streamlit application for QA:

```shell
# Run Streamlit app for QA
$ python agriqa/mqr/fred_rag_mqr_ques_ans.py
```

The configuration information is mentioned at *agriqa/mqr/fred_mqr_rag_config.py* file. Please adapt the config file according to your choice.


### Approaches
Some approaches are used for preliminary framework:
* *Embedding Model*: Pull **nomic-embed-text** model from Ollama website.
* *LLM*: Pull **gemma3** model from Ollama website.
* *Vector Store*: Chroma is used.
* *PDF Docs*: UnstructuredPDFLoader is used to parse PDFs. Document parsing is a crucial tasks. There are several ways to improve it by analysing different refinement techniques.
* *Chunking*: RecursiveCharacterTextSplitter for splitting.
* *Retriever*: **MultiQueryRetriever** with Maximal Marginal Relevance (MMR) is used. Currently analysing different approaches for multi-modality(for texts and tables), e.g., MultiVectorRetriever.
* Area of refinement approaches
    * PDF parsing, Retriever, Prompt, RAG chain etc. 