from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

load_dotenv()

llm = OpenAI(temperature=0)

text_splitter = CharacterTextSplitter()

with open('docs/modules/state_of_the_union.txt') as f:
    state_of_the_union = f.read()
texts = text_splitter.split_text(state_of_the_union)

print(texts)

docs = [Document(page_content=t) for t in texts[:3]]

print(docs)

chain = load_summarize_chain(llm, chain_type="map_reduce")
res = chain.run(docs)

print(res)
