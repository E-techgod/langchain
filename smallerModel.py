from llm import GROQ_LLM, GEMINI_LLM
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import AIMessage, HumanMessage, SystemMessage

basic_model = GEMINI_LLM
advanced_model = GROQ_LLM

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state['messages'])

    if message_count < 3: 
        model = basic_model
    else: 
        model = advanced_model

    request.model = model 

    return handler(request)

agent = create_agent(
    model = basic_model, 
    middleware = [dynamic_model_selection]
)

response = agent.invoke(
    {
        'messages' : [
            SystemMessage("You are a helpful asistant."),
            HumanMessage("What is 1+1?"),
            HumanMessage("What is PCA?"),
            HumanMessage("What is RPA?")
        ]
    }
)

print(response['messages'][-1].content)
print(response['messages'][-1].response_metadata['model_name'])