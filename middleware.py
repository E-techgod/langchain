
import os 
import time 

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-langsmith-api-key-here"
os.environ["LANGSMITH_PROJECT"] = "HooksDemo-Visualization"

from llm import llm 
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import HumanMessage, SystemMessage

load_dotenv(override=True)

class HooksDemo(AgentMiddleware): 

    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state : AgentState, runtime):
        self.start_time = time.time()
        print('before_agent triggered')

    def before_model(self, state : AgentState, runtime):
        print('before_model')

    def after_model(self, state : AgentState, runtime):
        print('after_model')

    def after_agent(self, state : AgentState, runtime):
        print('after_agent: ', time.time() - self.start_time)

agent = create_agent(
    model = llm, 
    middleware = [HooksDemo()]
)

# Prints the Mermaid markup code
# print(agent.get_graph().draw_mermaid())

response = agent.invoke({
    'messages' : [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("What is RPA?") 
    ]
})

print(response['messages'][-1].content)