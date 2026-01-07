from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and returns the result."""
    try:
        result = eval(expression)#  be careful with this because it's a security risk in real applications
    except Exception as e:
        return f"Error: {str(e)}"  
    
    return str(result) 

@tool("web_search_mock", return_direct=True)
def web_search_mock(query: str) -> str:
    """Return the capital of a given country if it exists in the mock data."""

    data = {
        "Brazil": "Brasilia", 
        "France": "Paris", 
        "Japan": "Tokyo"
    }

    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."    
    return f"I'm sorry, I don't know the capital of that country."

#llm = ChatOpenAI(model="gpt-5-nano", temperature=0, disable_streaming=True)
# Configure model
# chat_model = init_chat_model(
#     "gpt-5-mini",
#     temperature=0
# )
llm = ChatOpenAI(model="gpt-5-nano", temperature=0, disable_streaming=True)
tools = [web_search_mock]

# Define system prompt
SYSTEM_PROMPT = """
    Answer the following questions as best you can. You have access to the following tools:
    Only use the information you get from the tools, even if you know the answer.
    If the Information is not providd by the tools, say you don't know.

    {tools}

    Use the following format:
    Question: the input question you must answers
    Thought: you should always think about what to do
    Action: the action to take, should be one of the tools {tool_names}
    Action Input: the input to the action
    Observation: the result of the action
    
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original question
    
    Rules:
    - If you choose an Action, do NOT include Final Asnwer in the same step.
    - After Action and Action Input, stop and wait for the Observation.
    - Never search the internet. Ony use the tools provided.
    
    Begin!
    Question: {input}
    Thought: {agent_scratchpad}
    """


# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str



# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    country: str
    # Any interesting information about the weather if available
    capital: str 

# Set up memory
checkpointer = InMemorySaver()


#agent_chain = create_agent(llm, tools)
# Create agent
agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=agent_chain,
#     tools=tools,
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=5)

#print(agent.invoke({"input": "What is the capital of France?"}))
# print(agent_executor.invoke({"input": "What is the capital of Japan?"}))
# print(agent_executor.invoke({"input": "What is 23 multiplied by 47?"}))
# print(agent_executor.invoke({"input": "What is the capital of Brazil?"}))   


# # Run agent
# # `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"input": "What is the capital of France?"},
    config=config,
    context=Context(user_id="1")
)


print(response['structured_response'])
# # ResponseFormat(
# #     country="France",
# #     capital="Paris"
# # )


# # Note that we can continue the conversation using the same `thread_id`.
# response = agent.invoke(
#    {"input": "What is 23 multiplied by 47?"},
#     config=config,
#     context=Context(user_id="1")
# )

# print(response)
# # ResponseFormat(
# #     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
# #     weather_conditions=None
# # )
