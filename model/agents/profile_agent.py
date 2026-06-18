"""
profile_agent.py

Responsibilities:
1. Analyze competitive programming profile.
2. Identify strengths.
3. Identify weaknesses.
4. Generate profile summary.

It NEVER:
- recommends problems
- teaches concepts
- creates study plans
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from state.cpstate import CPCoachState


# =====================================================
# OUTPUT SCHEMA
# =====================================================

class ProfileAnalysis(BaseModel):

    strengths: list[str] = Field(
        description="Strong topics"
    )

    weaknesses: list[str] = Field(
        description="Weak topics"
    )

    summary: str = Field(
        description="Short profile summary"
    )


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

PROFILE_AGENT_PROMPT = """
You are the Profile Analysis Agent of CP Coach.

Your responsibilities:

1. Analyze competitive programming data.
2. Identify strengths.
3. Identify weaknesses.
4. Generate a concise summary.

You MUST NOT:

- Recommend problems
- Teach concepts
- Create study plans

Guidelines:

Strong Topic:
- High solve count
- High success rate
- Frequent activity

Weak Topic:
- Low solve count
- Poor contest performance
- Lack of practice

Return only structured output.
"""


# =====================================================
# PROMPT
# =====================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PROFILE_AGENT_PROMPT),
        (
            "human",
            """
            LeetCode Data:
            {leetcode_data}

            Codeforces Data:
            {codeforces_data}
            """
        )
    ]
)


# =====================================================
# CHAIN
# =====================================================

analysis_chain = (
    prompt
    | llm.with_structured_output(ProfileAnalysis)
)


# =====================================================
# NODE
# =====================================================

def profile_agent_node(
    state: CPCoachState
):
    """
    Input:
        state

    Reads:
        state["leetcode_data"]
        state["codeforces_data"]

    Writes:
        strengths
        weaknesses
        profile_summary
    """

    result = analysis_chain.invoke(
        {
            "leetcode_data": state["leetcode_data"],
            "codeforces_data": state["codeforces_data"]
        }
    )

    return {
        "strengths": result.strengths,
        "weaknesses": result.weaknesses,
        "profile_summary": result.summary
    }