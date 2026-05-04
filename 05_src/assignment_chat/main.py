import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START

from assignment_chat.prompts import get_system_prompt

load_dotenv(".env")
load_dotenv(".secrets")

_GATEWAY_URL = "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1"

chat_agent = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.3,
    openai_api_key="any_value",
    openai_api_base=_GATEWAY_URL,
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY", "")},
)


def call_model(state: MessagesState):
    response = chat_agent.invoke(
        [SystemMessage(content=get_system_prompt())] + state["messages"]
    )
    return {"messages": [response]}


def get_graph():
    builder = StateGraph(MessagesState)

    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.set_finish_point("call_model")

    return builder.compile()