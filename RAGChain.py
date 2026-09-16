import os 
from llm import llm 
from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
    model_kwargs={"device": "cpu"},         
    encode_kwargs={"normalize_embeddings": True}
)

texts = [
    'I love apples',
    'I love bananas',
    'I enjoy organes',
    'I think pears taste very good',
    'I hate mango',
    'I dislike raspberries',
    'I hate Windos',
    "I love Apple's products",
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

print(vector_store.similarity_search('What fruits does the person like', k=3))
print(vector_store.similarity_search('What fruits does the person dilike', k=3))

retriever = vector_store.as_retriever(search_kwargs={'k': len(texts)})

retriever_tool = create_retriever_tool(
    retriever, 
    name='kb_search', 
    description='Search the small product / fruit knowledge base for information')

agent = create_agent(
    model = llm, 
    tools = [retriever_tool],
    system_prompt = (
        """
        Your are helpful assistant. For questions about Macs, apples or laptops,
        first call the kb_search tool to retrieve context, then answer succintly. Maybe you have to use it multiple times before answering
        """
    )
)

result = agent.invoke(
    {
        "messages" : [
            {
                "role" : "user",
                "content" : "What three fruits does the person like and what are three fruits does the person dislike"
            }
        ]
    }
)

print("\nFinal answer:")
print(result["messages"][-1].content)