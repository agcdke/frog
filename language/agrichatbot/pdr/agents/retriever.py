'''
Retriever Agent: takes user’s question and searching the knowledge base for relevant information by vector search.
Vectors are computed using the nomic-embed-text model and stored in Chroma.
Prompt: suggest 3 search terms in English.
These terms are then concatenated with the original question and used to search for relevant information 
in the knowledge base.
'''

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from fred_rag_chatbot_config import fred_config


class RetrieverAgent:
    def __init__(self):
        self.llm = ChatOllama(model=fred_config["system1_model_name"], temperature=0)
        self.search_term_prompt_template = fred_config["search_term_prompt_template"]
        self.search_term_prompt = ChatPromptTemplate.from_template(
            self.search_term_prompt_template
        )

        self.search_term_chain = self.search_term_prompt | self.llm

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=50,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=25,
        )

        self.fs = LocalFileStore(fred_config["docstore_persist_directory"])
        self.store = create_kv_docstore(self.fs)

        self.vectorstore = Chroma(
            collection_name=fred_config["vector_store_name"],
            embedding_function=OllamaEmbeddings(model=fred_config["embedding_model"]),
            persist_directory=fred_config["chromadb_persist_directory"],
        )

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
            search_kwargs={"k": 3},
        )

    def run_search_term(self, question):
        result = self.search_term_chain.invoke({"question": question})
        search_terms = result.content
        search_text = f"{search_terms}, {question}"
        return search_text

    def invoke(self, state):
        question = state.get("question")
        search_text = self.run_search_term(question)
        relevant_docs = self.retriever.invoke(search_text)
        retrieved_info = ""
        for d in relevant_docs:
            retrieved_info += d.page_content
            retrieved_info += "\n"
        state["search_terms"] = search_text
        state["retrieved_info"] = retrieved_info

        return state
