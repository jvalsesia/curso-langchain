from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

print("OpenAI:")
openai = init_chat_model("gpt-5-nano", model_provider="openai", temperature=0.5)
answer_openai = openai.invoke("Hello world!")
print(answer_openai.content)

print("Gemini:")
gemini = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0.5)
answer_gemini = gemini.invoke("Hello world!")
print(answer_gemini.content)