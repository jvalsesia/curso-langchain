from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()

long_text = """
Beyond the Raider Stereotype
Viking culture, which dominated Scandinavia from the late 8th to the 11th century, 
is often misunderstood as merely a society of ruthless barbarians. 
While the image of the horned helmet—a 19th-century artistic invention—persists, 
the reality of the Norse people was far more complex and sophisticated. 
Originating from the rugged landscapes of modern-day Norway, Sweden, and Denmark, 
the majority of Vikings were actually peaceful farmers, fishermen, and master craftspeople. 
Their society was stratified but structured, divided into enslaved thralls, free peasants 
known as karls, and the aristocracy or jarls. Uniquely for the time, they operated under a 
legal system where free men gathered at an assembly called the Thing to settle disputes, 
representing an early form of democratic governance.

Engineering and Exploration
Central to their expansion was their unparalleled naval engineering. The iconic longship, 
with its shallow draft and symmetrical design, allowed them to cross open oceans to 
Iceland and North America while also navigating shallow rivers deep into Russia and France. 
This mobility turned them into history’s most prolific traders, creating networks that 
connected the Baltic directly to the Islamic Caliphates in Baghdad. Furthermore, 
Viking women enjoyed freedoms virtually unknown elsewhere in medieval Europe; they could 
own property, initiate divorce, and often managed the farm and finances while men were 
away on expeditions.

Mythology and Legacy
Their worldview was shaped by a fatalistic religion filled with giants, elves, and a pantheon
of gods led by Odin and Thor. They believed that the time of one's death was fated, and that 
dying with a weapon in hand was the only path to Valhalla, a belief that fueled their legendary 
fearlessness in battle. Ultimately, the Viking legacy is not just one of destruction, but of 
creation. They founded major cities like Dublin and Kyiv, influenced the English language with 
hundreds of words, and left behind a rich literary tradition of Sagas that continue to shape 
our understanding of the ancient world.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=70
)

parts = splitter.create_documents([long_text])

for part in parts:
    print(part.page_content)
    print("-"*30)

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

chain_summarize = load_summarize_chain(llm, chain_type="stuff", verbose=True)

result = chain_summarize.invoke({"input_documents": parts})
print("Resumo Final:")
print(result["output_text"])