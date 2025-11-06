import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# The Judge LLM (should be a powerful model for accurate scoring)
JUDGE_MODEL = ChatOllama( # <--- CHANGED
    model="llama3", 
    temperature=0 # Low temperature for reliable evaluation
)

def evaluate_hallucination_and_relevance(
    input_text: str, 
    model_output: str, 
    reference_context: str
) -> dict:
    """
    Uses an LLM (the Judge) to evaluate an LLM's output for Hallucination and Relevance.
    
    This is an example of a custom LLM-as-a-Judge evaluation.
    """
    
    # 1. Hallucination/Faithfulness Check Prompt
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI quality evaluator. Your task is to rate the 'Model Output' based on the 'Reference Context'. Rate the faithfulness (lack of hallucination) on a scale of 1 to 5. 5 is perfectly faithful, 1 is completely hallucinated. Return ONLY the score."),
        ("user", f"Model Output: {model_output}\n\nReference Context: {reference_context}")
    ])
    
    # 2. Relevance Check Prompt
    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI quality evaluator. Your task is to rate the 'Model Output' based on the original 'Input Query'. Rate the relevance on a scale of 1 to 5. 5 is perfectly relevant, 1 is completely irrelevant. Return ONLY the score."),
        ("user", f"Input Query: {input_text}\n\nModel Output: {model_output}")
    ])
    
    # Run the judge on both metrics
    try:
        faithfulness_score = JUDGE_MODEL.invoke(hallucination_prompt.format_messages(model_output=model_output, reference_context=reference_context)).content.strip()
        relevance_score = JUDGE_MODEL.invoke(relevance_prompt.format_messages(input_text=input_text, model_output=model_output)).content.strip()
        
        return {
            "faithfulness_score": int(faithfulness_score.split()[0]),
            "relevance_score": int(relevance_score.split()[0])
        }
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {"faithfulness_score": 0, "relevance_score": 0}