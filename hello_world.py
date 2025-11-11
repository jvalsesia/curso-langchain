import time
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def hello_world_openai():
    load_dotenv()
    start_time = time.perf_counter() 
    openai = ChatOpenAI(model_name="gpt-5-nano", temperature=0.5)
    answer_openai = openai.invoke("Hello world!")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(answer_openai)
    print(f"Elapsed time: {elapsed_time} seconds")