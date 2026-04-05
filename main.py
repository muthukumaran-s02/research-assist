import sys
from langchain_core.messages import HumanMessage
from agent import agent
from logger import logger

def run_agent(query: str):
    """Runs the LangGraph agent for research."""
    logger.info(f"--- Starting Research for: '{query}' ---\n")
    
    # Initialize state with user query
    initial_state = {"messages": [HumanMessage(content=query)]}
    
    # Run the graph
    for step in agent.stream(initial_state, stream_mode="values"):
        last_message = step["messages"][-1]
        
        # Display model responses
        if last_message.type == "ai" and not last_message.tool_calls:
            logger.info(f"\nFinal Answer:\n {last_message.content}")
        elif last_message.type == "tool":
            logger.info(f"[Tool Response] {last_message.name} output: {last_message.content[:100]}...")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your research query: ")
    
    run_agent(user_query)
