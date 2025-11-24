from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

system = ("system", "You are an assistant that answers questions in a {style} style!")
user = ("user", "{question}")

chat_prompt = ChatPromptTemplate.from_messages([system, user])
messages = chat_prompt.format_messages(style="friendly", question="What is LangChain?")

for message in messages:
    print(f"{message.type}: {message.content}")

model = ChatOpenAI(model_name="gpt-5-mini", temperature=0.5)
response = model.invoke(messages)
print("Response:", response.content)
