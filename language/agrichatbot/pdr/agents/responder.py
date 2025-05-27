'''
Responder Agent: final component that is responsible for replying to user.
Analyze GraphState and decide which LLM and prompt to use for generating the response.
For RAG prompt: essential to structure it with the following details to ensure accurate 
and context-aware responses and reduce hallucinations.
'''

from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from fred_rag_chatbot_config import fred_config

class ResponderAgent:
    def __init__(self):
        self.system1_model = ChatOllama(model="llama3.1", temperature=0)
        self.system2_model = ChatOllama(model="deepseek-r1:8b", temperature=0)
        self.responder_prompt_template = fred_config["responder_prompt_template"]
        self.responder_with_RAG_prompt_template = fred_config["responder_with_RAG_prompt_template"]

    def run(self, question, context="", req_think=False):
        if context == "":
            responder_prompt = ChatPromptTemplate.from_template(
                self.responder_prompt_template
            )
        else:
            responder_prompt = ChatPromptTemplate.from_template(
                self.responder_with_RAG_prompt_template
            )
        if not req_think:
            llm = self.system1_model
        else:
            llm = self.system2_model

        responder_chain = responder_prompt | llm
        if context == "":
            result = responder_chain.invoke({"question": question})
        else:
            result = responder_chain.invoke(
                {"question": question, "retrieved_info": context}
            )
        return result.content

    def invoke(self, state):
        question = state.get("question")
        if state.get("retrieved_info"):
            context = state.get("retrieved_info")
        else:
            context = ""
        responder_reply = self.run(
            question, context, state.get("router_need_system_2")
        )
        state["responder_reply"] = responder_reply
        return state