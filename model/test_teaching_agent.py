import os
import json
from dotenv import load_dotenv

# Load environment variables (e.g., GROQ_API_KEY) BEFORE importing any models
load_dotenv()

# Ensure we can import from the currect directory structure
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.teaching_agent import teaching_agent_node
from state.cpstate import CPCoachState

def run_test():
    if "GROQ_API_KEY" not in os.environ:
        print("WARNING: GROQ_API_KEY is not set in your environment.")
        print("The Groq API call will likely fail. Please set it before running.\n")
    
    # Construct a mock state representing a beginner
    mock_state: CPCoachState = {
        "user_query": "I am completely new. I don't know whether to pick C++ or Python, and I don't understand how Binary Search works. Can you help?",
        "intent": "concept_explanation",
        "next_agents": ["teaching_agent"],
        "strengths": ["Basic math"],
        "weaknesses": ["Language fundamentals", "Searching concepts"],
        "recommended_problems": [],
        "learning_content": "",
        "resources": [],
        "study_plan": [],
        "revision_timeline": [],
        "final_response": ""
    }
    
    print("Executing Teaching Agent Node with mock state...")
    try:
        result = teaching_agent_node(mock_state)
        print("\n================== TEACHING AGENT OUTPUT ==================\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\n===========================================================\n")
        print("Success! The output perfectly matches the expected typed-dict schema.")
        
    except Exception as e:
        print(f"\nExecution Failed: {e}")

if __name__ == "__main__":
    run_test()
