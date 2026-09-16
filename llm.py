import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GROQ_LLM = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for the Groq fallback")

GEMINI_LLM = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0.0,
    google_api_key=gemini_api_key,
)

llm = GROQ_LLM.with_fallbacks(
    [GEMINI_LLM],
)