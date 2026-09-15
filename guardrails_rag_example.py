from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from guardrails import Guard
from guardrails.hub import RAGAsContextRelevance  # Example context relevance validator

# 1. Setup Guardrails AI validator (Semantic Layer)
# This checks if the retrieved context actually relates to the user's question
guard = Guard().use(
    RAGAsContextRelevance(on_fail="fix") # Fallback behavior if relevance fails
)

# 2. Setup VDB Retriever with your 0.35 threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.35, "k": 3}
)

def retrieve_and_validate(inputs):
    query = inputs["question"]
    docs = retriever.invoke(query)
    
    # --- LAYER 1: LCEL Structural Check (Deterministic) ---
    # If the VDB returns nothing, short-circuit immediately.
    if not docs:
        return {
            "context": "",
            "question": query,
            "answer": "I cannot answer this based on the available information."
        }
    
    context_text = "\n\n".join(doc.page_content for doc in docs)
    
    # --- LAYER 2: Guardrails AI Check (Semantic) ---
    # Run the guardrail to verify context-to-query relevance
    validation_result = guard.validate(
        context_text,
        metadata={"query": query}
    )
    
    if not validation_result.validation_passed:
        return {
            "context": "",
            "question": query,
            "answer": "I cannot answer this based on the available information."
        }
        
    return {
        "context": context_text,
        "question": query,
        "answer": None # Signals it's safe to proceed to LLM generation
    }

# 3. Define prompt and LLM generation step
prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY this context:\n{context}\n\nQuestion:\n{question}"
)

# 4. Construct the full LCEL Chain
chain = (
    RunnablePassthrough()
    | RunnableLambda(retrieve_and_validate)
    | (lambda x: x["answer"] if x["answer"] else (
        {"context": x["context"], "question": x["question"]} 
        | prompt 
        | llm 
        | StrOutputParser()
    ))
)