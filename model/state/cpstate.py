from typing import TypedDict, List

class CPCoachState(TypedDict):

    # user input
    user_query: str

    # decided by orchestrator
    intent: str

    # which agents should run
    next_agents: List[str]

    # profile analysis
    profile_summary: str
    strengths: List[str]
    weaknesses: List[str]
    
    # progression state (UFP personalized data)
    solved_problems: List[str]      # list of LC titleSlugs the user has already solved
    topic_mastery: dict             # nested dict showing performance (e.g. {"Dynamic Programming": {"level": "Beginner", "easy_solved": 3}})

    # recommendations
    focus_topics: List[str]
    recommended_problems: list
    recommendation_reasoning: str

    # teaching
    learning_content: str
    resources: list
    study_plan: List[str]
    revision_timeline: List[dict]

    # final answer
    final_response: str