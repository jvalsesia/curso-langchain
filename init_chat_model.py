import time
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

def hello_world_gemini():
    load_dotenv()
    start_time = time.perf_counter() 
    gemini = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0.5)
    answer_gemini = gemini.invoke("Hello world!")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(answer_gemini)
    print(f"Elapsed time: {elapsed_time} seconds")