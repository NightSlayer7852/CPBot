"""
orchestrator.py

The orchestrator is the brain of the CP Coach.

Responsibilities:
1. Understand user intent.
2. Decide which agent(s) should run.
3. Update graph state.

It NEVER:
- teaches concepts
- analyzes profiles
- recommends problems
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from state.state import CPCoachState


# =====================================================
# INTENTS
# =====================================================

class Intent(str, Enum):
    PROFILE_ANALYSIS = "profile_analysis"
    PROBLEM_RECOMMENDATION = "problem_recommendation"
    CONCEPT_EXPLANATION = "concept_explanation"
    STUDY_PLAN = "study_plan"
    BEGINNER_GUIDANCE = "beginner_guidance"


# =====================================================
# STRUCTURED OUTPUT
# =====================================================

class RoutingDecision(BaseModel):
    """
    Output schema for the orchestrator.
    """

    intent: Intent = Field(
        description="Detected user intent"
    )

    next_agents: List[str] = Field(
        description="Agents that should execute next"
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

ORCHESTRATOR_PROMPT = """
You are the Orchestrator Agent of CP Coach.

Your ONLY responsibility:

1. Understand the user's request.
2. Decide which agents should run.
3. Return valid structured output.

Available Agents:

profile_agent
recommendation_agent
teaching_agent

Routing Rules:

PROFILE ANALYSIS
Examples:
- Analyze my profile
- What are my weaknesses?
- Analyze my Codeforces account

Agents:
profile_agent


CONCEPT EXPLANATION
Examples:
- Explain DP
- Teach Graphs
- What is Binary Search?

Agents:
teaching_agent


PROBLEM RECOMMENDATION
Examples:
- What should I solve next?
- Recommend problems
- Give me practice questions

Agents:
profile_agent
recommendation_agent


STUDY PLAN
Examples:
- What should I study today?
- Create today's plan
- What should I focus on?

Agents:
profile_agent
teaching_agent


BEGINNER GUIDANCE
Examples:
- I am a beginner
- How do I start CP?
- Which language should I learn?

Agents:
teaching_agent

Never answer the user.

Only decide routing.
"""


# =====================================================
# PROMPT
# =====================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ORCHESTRATOR_PROMPT),
        ("human", "{query}")
    ]
)


# =====================================================
# CHAIN
# =====================================================

routing_chain = (
    prompt
    | llm.with_structured_output(RoutingDecision)
)


# =====================================================
# NODE
# =====================================================

def orchestrator_node(
    state: CPCoachState
):
    """
    LangGraph node.

    Input:
        state

    Output:
        {
            "intent": "...",
            "next_agents": [...]
        }
    """

    query = state["user_query"]

    decision = routing_chain.invoke(
        {
            "query": query
        }
    )

    return {
        "intent": decision.intent.value,
        "next_agents": decision.next_agents
    }