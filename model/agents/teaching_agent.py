"""
teaching_agent.py

Responsibilities:
1. Explain concepts.
2. Recommend resources.
3. Create beginner roadmaps.
4. Generate study plans.

It NEVER:
- analyzes profiles
- recommends problems
- routes the graph
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from state.state import CPCoachState


# =====================================================
# OUTPUT SCHEMA
# =====================================================

class TeachingOutput(BaseModel):

    learning_content: str = Field(
        description="Concept explanation"
    )

    resources: list[str] = Field(
        description="Recommended resources"
    )

    study_plan: list[str] = Field(
        description="Short study plan"
    )


# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)


# =====================================================
# SYSTEM PROMPT
# =====================================================

TEACHING_AGENT_PROMPT = """
You are the Teaching Agent of CP Coach.

Your responsibilities:

1. Explain concepts.
2. Help beginners.
3. Recommend resources.
4. Create study plans.

Rules:

- Be concise.
- Be beginner friendly.
- Explain step-by-step.
- Use simple language.
- Do not recommend problems.
- Do not analyze profiles.

For concept explanations:

Explain:
1. What it is
2. Why it is needed
3. Intuition
4. When to use it

For beginners:

Recommend:
- Language
- Learning order
- Resources

For study plans:

Generate a practical plan.
Maximum 5 steps.

Return structured output only.
"""


# =====================================================
# PROMPT
# =====================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", TEACHING_AGENT_PROMPT),
        (
            "human",
            """
            User Query:
            {query}

            Strengths:
            {strengths}

            Weaknesses:
            {weaknesses}
            """
        )
    ]
)


# =====================================================
# CHAIN
# =====================================================

teaching_chain = (
    prompt
    | llm.with_structured_output(
        TeachingOutput
    )
)


# =====================================================
# NODE
# =====================================================

def teaching_agent_node(
    state: CPCoachState
):
    """
    Reads:
        user_query
        strengths
        weaknesses

    Writes:
        learning_content
        resources
        study_plan
    """

    result = teaching_chain.invoke(
        {
            "query": state["user_query"],
            "strengths": state.get("strengths", []),
            "weaknesses": state.get("weaknesses", [])
        }
    )

    return {
        "learning_content": result.learning_content,
        "resources": result.resources,
        "study_plan": result.study_plan
    }