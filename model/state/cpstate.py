
class CPCoachState(TypedDict):

    # user input
    user_query: str

    # decided by orchestrator
    intent: str

    # which agents should run
    next_agents: list[str]

    # profile analysis
    strengths: list[str]
    weaknesses: list[str]

    # recommendations
    recommended_problems: list

    # teaching
    learning_content: str
    resources: list

    # final answer
    final_response: str