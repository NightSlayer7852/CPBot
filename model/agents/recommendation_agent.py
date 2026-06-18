"""
recommendation_agent.py

Responsibilities:
1. Recommend next problems.
2. Recommend focus topics.
3. Explain WHY those problems were selected.

It NEVER:
- Analyze profiles
- Teach concepts
- Create learning roadmaps
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from state.cpstate import CPCoachState


# =====================================================
# OUTPUT SCHEMA
# =====================================================

class RecommendedProblem(BaseModel):

    platform: str = Field(
        description="leetcode or codeforces"
    )

    title: str = Field(
        description="Problem name"
    )

    topic: str = Field(
        description="Main topic"
    )

    difficulty: str = Field(
        description="easy, medium, hard"
    )


class RecommendationOutput(BaseModel):

    focus_topics: list[str]

    recommended_problems: list[RecommendedProblem]

    reasoning: str


# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# =====================================================
# SYSTEM PROMPT
# =====================================================

RECOMMENDATION_PROMPT = """
You are the Recommendation Agent of CP Coach.

Your responsibilities:

1. Recommend next problems.
2. Select the most important topics.
3. Explain why those topics matter.

You MUST use:

- strengths
- weaknesses
- profile summary

Rules:

1. Focus on weaknesses first.
2. Do not recommend extremely difficult problems.
3. Recommend gradual progression.
4. Maximum 5 problems.
5. Return structured output only.

Example:

Weakness:
Dynamic Programming

Recommendation:

1. Climbing Stairs
2. House Robber
3. Coin Change

Reason:
Build DP fundamentals before advanced DP.
"""


# =====================================================
# PROMPT
# =====================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RECOMMENDATION_PROMPT),
        (
            "human",
            """
            Strengths:
            {strengths}

            Weaknesses:
            {weaknesses}

            Profile Summary:
            {profile_summary}
            """
        )
    ]
)


# =====================================================
# CHAIN
# =====================================================

recommendation_chain = (
    prompt
    | llm.with_structured_output(
        RecommendationOutput
    )
)


# =====================================================
# NODE
# =====================================================

def recommendation_agent_node(
    state: CPCoachState
):
    """
    Reads:
        strengths
        weaknesses
        profile_summary

    Writes:
        focus_topics
        recommended_problems
        recommendation_reasoning
    """

    result = recommendation_chain.invoke(
        {
            "strengths": state["strengths"],
            "weaknesses": state["weaknesses"],
            "profile_summary": state["profile_summary"]
        }
    )

    return {
        "focus_topics": result.focus_topics,
        "recommended_problems": [
            problem.model_dump()
            for problem in result.recommended_problems
        ],
        "recommendation_reasoning": result.reasoning
    }