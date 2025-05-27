fred_config = dict(
    prefer_pull_model = False,
    doc_dir_path = "data/pdf/unece_product_ standards/",
    model_name = "gemma3",
    embedding_model = "nomic-embed-text",
    vector_store_name = "fred-rag",
    chromadb_persist_directory = "artifacts/mqr_chroma_db",
    fred_contextual_promt_template = """
    You are a LLM assistant for FRED project. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the search resukts. 
    If you are unsure or do not know the answer to a user's question, 
    do not guess or invent information. Instead, clearly state that you do not know, 
    and then attempt to retrieve relevant information from external sources or your 
    knowledge base before responding. Provide these alternative questions separated by newlines.
    Original question: {question}""",
    rag_prompt_template = """
    Answer the question based only on the following context:
    {context}
    Question: {question} 
    """,
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is.
    """,
    system_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer
    the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the
    answer concise.
    \n\n
    {context}
    """,

)