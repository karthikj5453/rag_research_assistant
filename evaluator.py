import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from rag_pipeline import ask
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Note: Ragas defaults to OpenAI. To use Gemini, we need to wrap it.
# For this demonstration, we'll set up a mock evaluation loop or instruction
# on how to plug in Gemini as the evaluator.

def run_evaluation(test_questions, ground_truths):
    """
    Runs Ragas evaluation on a set of test questions.
    """
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": ground_truths
    }

    print(f"🚀 Running evaluation on {len(test_questions)} questions...")

    for query in test_questions:
        answer, context_items = ask(query)
        
        data["question"].append(query)
        data["answer"].append(answer)
        # Ragas expects a list of strings for contexts
        data["contexts"].append([item["text"] for item in context_items])

    # Create dataset
    dataset = Dataset.from_dict(data)

    # Note: To use Gemini with Ragas, you would normally pass an 'llm' parameter
    # results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall], llm=gemini_llm)
    
    # For now, we'll try to run it with default (OpenAI) but warn if missing.
    # If the user doesn't have OPENAI_API_KEY, this might fail.
    # In a real production setup, we'd configure LangChain Gemini here.
    
    try:
        print("📊 Computing metrics...")
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall]
        )
        return result
    except Exception as e:
        print(f"⚠️ Evaluation failed (likely due to missing Evaluator LLM API Key): {e}")
        # Return the collected data as a dataframe for manual review
        return pd.DataFrame(data)

if __name__ == "__main__":
    # Example test set
    test_set = [
        "What is the main contribution of the paper?",
        "What methodology was used in the research?",
        "What are the key findings or results?"
    ]
    # In a real scenario, you'd provide actual ground truth answers from the paper
    ground_truths = [
        "The primary contribution is [Insert based on paper].",
        "The study utilized [Insert methodology].",
        "The results demonstrated [Insert findings]."
    ]

    results = run_evaluation(test_set, ground_truths)
    print("\n--- EVALUATION RESULTS ---")
    print(results)
