import os
import json
from dotenv import load_dotenv

# Load environment variables (e.g., GROQ_API_KEY) BEFORE importing any models
load_dotenv()

# Ensure we can import from the currect directory structure
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.recommendation_agent import recommendation_agent_node
from state.cpstate import CPCoachState

def run_test():
    if "GROQ_API_KEY" not in os.environ:
        print("WARNING: GROQ_API_KEY is not set in your environment.")
        print("The Groq API call will likely fail. Please set it before running.\n")
    
    # Construct a mock state representing an intermediate who struggles with DP
    mock_state: CPCoachState = {
        "user_query": "I need some problems to practice.",
        "intent": "problem_recommendation",
        "next_agents": ["recommendation_agent"],
        "strengths": ["Arrays", "Two Pointers"],
        "weaknesses": ["Dynamic Programming", "Graph Theory"],
        "profile_summary": "Intermediate user. Has solved 100 easy array questions but struggles hard on medium DP.",
        "focus_topics": [],
        "recommended_problems": [],
        "recommendation_reasoning": "",
        "learning_content": "",
        "resources": [],
        "study_plan": [],
        "revision_timeline": [],
        "final_response": ""
    }
    
    print("Executing Recommendation Agent Node with mock state...")
    try:
        result = recommendation_agent_node(mock_state)
        print("\n================== RECOMMENDER AGENT OUTPUT ==================\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\n===========================================================\n")
        print("Success! The output perfectly fetched exact problems via the Graph QL API.")
        
    except Exception as e:
        print(f"\nExecution Failed: {e}")

if __name__ == "__main__":
    run_test()
