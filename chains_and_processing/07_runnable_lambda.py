from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()   

def parse_number(text: str) -> int: 
    return int(text.strip())
  
"""
A runnable that extracts a number from text.
Can be used in a chain to process LLM outputs.
"""
parse_runnable = RunnableLambda(parse_number)

number = parse_runnable.invoke("42")
print(f"Extracted number: {number}")