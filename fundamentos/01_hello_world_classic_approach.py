from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

print("OpenAI:")
openai_llm = ChatOpenAI(model="gpt-5-nano", temperature=0.5)
answer_openai = openai_llm.invoke("Hello world!")
print(answer_openai.content)


print("Gemini:")
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
answer_gemini = gemini_llm.invoke("Hello world!")
print(answer_gemini.content)