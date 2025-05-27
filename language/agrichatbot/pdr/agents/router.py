'''
Router Agent: determines whether a user query (a) Requires RAG or (b) Requires System-2 thinking.
Structure the output as JSON: use an LLM capable of function calling.
'''
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from fred_rag_chatbot_config import fred_config

# Basemodels models help validate and structure the output data from the LLM, ensuring it can be seamlessly 
# parsed and used within the graph
class RequireRetrieval(BaseModel):
    requires_retrieval: bool
    reason: str


class RequireThinking(BaseModel):
    requires_thinking: bool
    reason: str

class RouterAgent:
    def __init__(self):
        self.llm = ChatOllama(model=fred_config["system1_model_name"], temperature=0)

        self.req_ret_output_parser = PydanticOutputParser(
            pydantic_object=RequireRetrieval
        )
        self.req_thi_output_parser = PydanticOutputParser(
            pydantic_object=RequireThinking
        )

        self.req_ret_prompt_template = fred_config["req_ret_prompt_template"]

        self.req_thi_prompt_template = fred_config["req_thi_prompt_template"]

        self.req_ret_prompt = ChatPromptTemplate.from_template(
            self.req_ret_prompt_template
        )
        self.req_ret_chain = (
            self.req_ret_prompt | self.llm | self.req_ret_output_parser
        )

        self.req_thi_prompt = ChatPromptTemplate.from_template(
            self.req_thi_prompt_template
        )
        self.req_thi_chain = (
            self.req_thi_prompt | self.llm | self.req_thi_output_parser
        )

    def run(self, question):
        ret_result = self.req_ret_chain.invoke(
            {
                "question": question,
            }
        )

        thi_result = self.req_thi_chain.invoke(
            {
                "question": question,
            }
        )
        return ret_result, thi_result

    def invoke(self, state):
        question = state.get("question")
        try:  # sometimes it will fail due to guardrails
            ret_result, thi_result = self.run(question)

            state["router_need_retriever"] = ret_result.requires_retrieval
            state["router_need_system_2"] = thi_result.requires_thinking
            state["router_need_retriever_reason"] = ret_result.reason
            state["router_need_system_2_reason"] = thi_result.reason
            
        except Exception as e:
            print(e)
            state["router_need_retriever"] = False
            state["router_need_system_2"] = False

        return state
