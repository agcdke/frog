fred_config = dict(
    prefer_pull_model = False,
    doc_dir_path = "data/pdf/GS1Test/",
    system1_model_name = "llama3.1",
    system2_model_name = "deepseek-r1",
    embedding_model = "nomic-embed-text",
    vector_store_name = "fred-rag-chat",
    chromadb_persist_directory = "artifacts/chat_chroma_db",
    docstore_persist_directory = "artifacts/chat_docstore",
    req_ret_prompt_template = """
    # Task
    You are an intelligent routing agent. Given a user question determine if the question requires:
    - "knowledge retrieval on FRED". If the question is asking about FRED, return True
    
    Strictly Return your response in this JSON format:
    {{"requires_retrieval": "<bool>", "reason": "<explanation>"}}
    
    Question: {question}
    """,
    req_thi_prompt_template="""
    # Task
    You are an intelligent routing agent. Given a user question determine if the question requires:
    - "thinking". If the question is complicated and need some thinking before you can reply, return True

    Strictly Return your response in this JSON format:
    {{"requires_thinking": "<bool>", "reason": "<explanation>"}}

    Question: {question}
    """,
    search_term_prompt_template="""
    # Task
    You are an intelligent search term suggestion agent. Given a user question, suggest search english terms, 
    up to 3 words, which will optimize the vector search. Strictly Return your response with just the search term.

    Question: {question}
    """,
    responder_prompt_template="""
    # Task
    You are an intelligent responder agent. Please provide your response in a clear, concise, and 
    professional manner. Use formal language, maintain a respectful tone, and ensure your answer is 
    well-structured and informative. Reply in whatever language the question is in.

    Question: {question}
    """,
    responder_with_RAG_prompt_template="""
    # Task
    You are an intelligent responder agent. Given a user question, and retrieved context, strictly follow the 
    information given the retrieved context. If you are unsure or do not know the answer to a user's question, 
    do not guess or invent information. Instead, clearly state that you do not know.
    Please provide your response in a clear, concise, and professional manner. Use formal language, maintain 
    a respectful tone, and ensure your answer is well-structured and informative. Reply in whatever language 
    the question is in.

    Retrieved Context: {retrieved_info}
    Question: {question}
    """,

)