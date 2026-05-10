import gradio as gr
from dotenv import load_dotenv
from utils.logger import get_logger

from langchain_core.messages import HumanMessage, AIMessage

from assignment_chat.main import get_graph

load_dotenv(".env")
load_dotenv(".secrets")
_logs = get_logger(__name__)


chatbot = get_graph()


def chat(message: str, history: list[dict]) -> str:
    # check for restricted topics
    blocked_terms = ["cat", "dog", "zodiac", "horoscope", "taylor swift"]
    if any(term in message.lower() for term in blocked_terms):
        return "I'm not able to discuss that topic."

    _logs.info(f"[ USER ]: {message}")

    langchain_messages = []

    for msg in history:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))

    langchain_messages.append(HumanMessage(content=message))

    state = {
        "messages": langchain_messages
    }
    response = chatbot.invoke(state)
    _logs.info(f"[ BOT  ]: {response['messages'][-1].content[:100]}...")

    return response["messages"][-1].content


chat_ui = gr.ChatInterface(
    fn=chat,
    type="messages"
)


if __name__ == "__main__":
    _logs.info("Starting chatbot UI ...")
    chat_ui.launch()