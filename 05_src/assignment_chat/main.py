import os
from dotenv import load_dotenv
from utils.logger import get_logger
import requests
import xml.etree.ElementTree as ET
import time 

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.tools import tool

from assignment_chat.prompts import get_system_prompt
# from assignment_chat.tools_arxiv import make_arxiv_tool

load_dotenv(".env")
load_dotenv(".secrets")

_logs = get_logger(__name__)

# hardcoded variables (typically would go in .env)
_DEBUG = False
if _DEBUG:
    _MAX_TOKENS = 50
    _MAX_RETRIES = 1
    _MAX_MESSAGE_HISTORY = 50
else:
    _MAX_TOKENS = 500
    _MAX_RETRIES = 2
    _MAX_MESSAGE_HISTORY = 20

_GATEWAY_URL = "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1"


chat_agent = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.3,
    max_tokens=_MAX_TOKENS,
    max_retries=_MAX_RETRIES,    
    openai_api_key="any_value",
    openai_api_base=_GATEWAY_URL,
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY", "")},
)


# tool placeholder
@tool
def get_current_time() -> str:
    """
    Returns the current date and time.
    Use this when the user asks what time or date it is.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_papers(query: str) -> str:
    """
    Search arXiv and return raw paper titles and abstracts.
    """
    base_url = "http://export.arxiv.org/api/query"
    
    headers = {"User-Agent": "vision-research-bot/1.0"}
    
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": 3
    }

    for attempt in range(3):
        response = requests.get(base_url, params=params, headers=headers)

        if response.status_code == 200:
            break
        elif response.status_code == 429:
            time.sleep(2 ** attempt)
        else:
            return f"API error {response.status_code}"
    else:
        return "Failed after retries"

    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    output = []

    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()

        title = " ".join(title.split())
        summary = " ".join(summary.split())

        output.append(
            f"Title: {title}\nSummary: {summary[:300]}...\n"
        )

    return "\n\n".join(output)

tools = [get_current_time, search_papers]


def call_model(state: MessagesState):
    """
    LLM decides whether to call tool or not
    and manages memory with sliding window
    """
    # memory management
    messages = state["messages"]
    if len(messages) > _MAX_MESSAGE_HISTORY:
        messages = messages[-_MAX_MESSAGE_HISTORY:]
    
    # call model
    response = chat_agent.bind_tools(tools).invoke(
        [SystemMessage(content=get_system_prompt())] + messages
    )
    return {"messages": [response]}


def get_graph():
    builder = StateGraph(MessagesState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "call_model")

    # add tool routing if tools exist
    if tools:
        builder.add_conditional_edges("call_model", tools_condition)
        builder.add_edge("tools", "call_model")
    else:
        builder.set_finish_point("call_model")

    return builder.compile()