# Install langgraph if not already installed
# pip install langgraph


from langchain.agents import create_agent
# create_react_agent has been moved to `langchain.agents`. 
# Please update your import to `from langchain.agents import create_agent`. 
# Deprecated in LangGraph V1.0 to be removed in V2.0.

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv()



@tool
def get_product_price(product_name: str) -> str:
    """Get the price of a product from the database."""
    prices = {
        "laptop": "$999",
        "mouse": "$25",
        "keyboard": "$75",
        "monitor": "$350"
    }
    return prices.get(product_name.lower(), "Product not found in database")

@tool
def get_product_stock(product_name: str) -> str:
    """Check if a product is in stock."""
    stock = {
        "laptop": "5 units available",
        "mouse": "50 units available",
        "keyboard": "0 units - out of stock",
        "monitor": "12 units available"
    }
    return stock.get(product_name.lower(), "Product not found in database")

tools = [get_product_price, get_product_stock]
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create agent using LangGraph
# agent_executor = create_react_agent(llm, tools)
agent_executor = create_agent(llm, tools)
# Run the agent
response = agent_executor.invoke({
    "messages": [("user", "What is the price of a laptop?")]
})

print("Response:", response)
print("\nMessages:")
for msg in response['messages']:
    print(f"{msg.type}: {msg.content}")