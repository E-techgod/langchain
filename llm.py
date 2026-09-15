import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)