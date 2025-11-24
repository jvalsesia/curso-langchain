from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["product"],
    template="Give me a creative name for a product that is a {product}."
)

text = template.format(product="smartphone with a foldable screen")
print(text)