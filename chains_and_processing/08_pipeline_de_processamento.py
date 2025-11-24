from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to French:\n {initial_text}"
)

template_summary = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 words:\n {text}"
)

llm_fr = ChatOpenAI(model_name="gpt-5-mini", temperature=0.5)

# pega o resultado do llm e tranforma numa string pronto para ser parseado em formato de runnable
translate = template_translate | llm_fr | StrOutputParser()

pipeline = {"text": translate} | template_summary | llm_fr | StrOutputParser()

result = pipeline.invoke({"initial_text": "Hello world! This is a test of the LangChain pipeline processing."})
print("Final Result:", result)