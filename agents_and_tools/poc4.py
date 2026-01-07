"""
The function "create_react_agent" is deprecated
  create_react_agent has been moved to `langchain.agents`. Please update your import to `from langchain.agents import create_agent
"""

#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool



from dotenv import load_dotenv

load_dotenv()


# 1. Define your tools
@tool
def get_weather(city: str):
    """Get the current weather for a city."""
    if city.lower() == "san francisco":
        return "It's 65°F and foggy."
    return "Sunny and 75°F."

tools = [get_weather]
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Create the agent
# This returns a compiled LangGraph graph
agent = create_agent(model, tools)

# 3. Use it
inputs = {"messages": [("user", "What is the weather in San Francisco?")]}
for chunk in agent.stream(inputs, stream_mode="values"):
    chunk["messages"][-1].pretty_print()