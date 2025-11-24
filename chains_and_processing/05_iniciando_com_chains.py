from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

question_template = PromptTemplate(
    input_variables=["name"],
    template="Hi, I'm {name}! Tell me a joke with my name!"
)

model = ChatOpenAI(model_name="gpt-5-mini", temperature=0.5)

# chain creation
# saida do template e vai entrar no model
chain = question_template | model

result = chain.invoke({"name": "Julio"})
print(result.content)

