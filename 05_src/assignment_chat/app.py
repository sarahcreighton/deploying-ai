import gradio as gr
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage

from assignment_chat.main import get_graph

load_dotenv(".secrets")


chatbot = get_graph()


def chat(message: str, history: list[dict]) -> str:
    
    langchain_messages = []

    for msg in history:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
    
    langchain_messages.append(HumanMessage(content=message))

    state = {
        "messages": langchain_messages,
    }
    response = chatbot.invoke(state)
    return response["messages"][-1].content


chat_ui = gr.ChatInterface(
    fn=chat,
    type="messages"
)


if __name__ == "__main__":
    chat_ui.launch()