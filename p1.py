from pydantic import BaseModel

import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import MemorySaver 
from langchain.agents.structured_output import ToolStrategy

class Context(BaseModel):
    user_id : str

class ResponseFormat(BaseModel): 
    summary : str
    temp_c : float
    temp_f : float
    humidity : float

@tool("locate_user", description="Look up user's city from context")
def locate_user(runtime: ToolRuntime[Context]) -> str:
    match runtime.context.user_id:
        case "ABC123":
            return "Vienna"
        case "XYZ456":
            return "London"
        case 'DEFG123':
            return 'Paris'
        case _:
            return 'Unknow' 

@tool("get_weather", description="Return current weather summary for a given city", return_direct= False) # Default is set to False
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
        "temp_c": float(current.get("temp_C")),
        "temp_f": float(current.get("temp_F")),
        "condition": weather_desc,
        "humidity": float(current.get("humidity")),
        "wind_mph": float(current.get("windspeedMiles")),
    }

load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0.0,
    groq_api_key = os.getenv("GROQ_API_KEY"),
)

checkpointer = MemorySaver()

agent = create_agent(
    model = llm,
    tools = [get_weather, locate_user],
    system_prompt = 'You are a helpful weather assistant',
    context_schema = Context,
    checkpointer = checkpointer
)

config = {
    'configurable' : {'thread_id' : 1}
}

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather like?"}
        ]
    },
    config=config,
    context=Context(user_id="DEFG123"),
)

print(response["messages"][-1].content)

config = {
    'configurable' : {'thread_id' : 1}
}

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Amd is this usual?"}
        ]
    },
    config=config,
    context=Context(user_id="DEFG123"),
)

print(response["messages"][-1].content)