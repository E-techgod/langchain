from dataclasses import dataclass

import os
import requests
from dotenv import load_dotenv

from langchain.tools import ToolRuntime, tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    user_id : str

@dataclass
class ResponseFortmat: 
    summary : str
    temp_c : float
    temp_f : float
    humidity : float

@tool('locate_user', description= "Look up a user's city based on the context", return_direct= False)
def locate_user (runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Vienna'
        case 'XYZ456':
            return 'London'
        case 'HJKL789':
            return 'Paris'
        case _:
            return 'Unknown' 


@tool("get_weather", description="Return current weather summary for a given city", return_direct= True) # Default is set to False
def get_weather(city: str) -> dict:
    url = f"https://wttr.in/{city}?format=j1"
    data = requests.get(url, timeout=10).json()

    current = data.get("current_condition", [{}])[0]
    weather_desc = (
        current.get("weatherDesc", [{}])[0].get("value")
        if current.get("weatherDesc")
        else "Unknown"
    )

    return {
        "city": city,
        "temp_c": current.get("temp_C"),
        "temp_f": current.get("temp_F"),
        "condition": weather_desc,
        "humidity_pct": current.get("humidity"),
        "wind_mph": current.get("windspeedMiles"),
    }

load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0.0,
    groq_api_key = os.getenv("GROQ_API_KEY"),
)

agent = create_agent(
    model = llm,
    tools = [get_weather],
    system_prompt = 'You are a helpful weather assistant'
)

response = agent.invoke({
    'messages' : [
        {'role' : 'user', 'content' : "What is the weather in Vienna?"}
    ]
})

print(response)
print(response['messages'][-1].content)