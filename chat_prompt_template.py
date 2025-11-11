from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


def get_chat_prompt():
    load_dotenv()
    system = ("system", "You are an assistant that answers question in a {stype} style.")
    user = ("user", "{question}")
    chat_prompt = ChatPromptTemplate.from_messages([system, user])
    prompt = chat_prompt.format_prompt(stype="humorous", question="Tell me a joke")

    model = ChatOpenAI(model_name="gpt-5-mini", temperature=0.5)
    response = model.invoke(prompt.to_messages())
    print(response.content)
    