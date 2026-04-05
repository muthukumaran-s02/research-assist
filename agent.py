from typing import Annotated, TypedDict, List, Union
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from tools import tools

# Define the state of the graph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Initialize the model with tools
model = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0
).bind_tools(tools)

# Define the nodes
def call_model(state: AgentState):
    """Invokes the model with the current state."""
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Determines whether to continue or stop based on the last message."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Define edges
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Compile the graph
agent = workflow.compile()
