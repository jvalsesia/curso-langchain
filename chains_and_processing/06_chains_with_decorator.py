from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from dotenv import load_dotenv


load_dotenv()

@chain
def square(input_dict: dict) -> dict:
    x = input_dict["x"]
    return {"squared_result": x * x}


question_template = PromptTemplate(
    input_variables=["name"],
    template="Hi, I'm {name}! Tell me a joke with my name!"
)


question_template_2 = PromptTemplate(
    input_variables=["squared_result"],
    template="Tell me about the number {squared_result}"
)

model = ChatOpenAI(model_name="gpt-5-mini", temperature=0.5)

# chain creation
# saida do template e vai entrar no model
chain = question_template | model


chain2 = square | question_template_2 | model

result = chain.invoke({"name": "Julio"})
print(result.content)

result2 = chain2.invoke({"x": 5})
print(result2.content)