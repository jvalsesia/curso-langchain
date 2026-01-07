from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory



from dotenv import load_dotenv

load_dotenv()


prompt = ChatPromptTemplate.from_messages([
    "system", "You are a helpful assistant that stores and retrieves historical chat messages.",
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chat_model = ChatOpenAI(model="gpt-5-nano", temperature=0.8, disable_streaming=True)

chain = prompt | chat_model

session_storage: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_storage:
        session_storage[session_id] = InMemoryChatMessageHistory()
    return session_storage[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "dfyaod"}}

reponse1 = conversational_chain.invoke({"input": "Hello, who won the world series in 2020?"}, config=config)
print("Assistant Response 1:", reponse1.content)

reponse2 = conversational_chain.invoke({"input": "Where was it played?"}, config=config)
print("Assistant Response 2:", reponse2.content)

response3 = conversational_chain.invoke({"input": "Who was the MVP?"}, config=config)
print("Assistant Response 3:", response3.content)