"""
teaching_agent.py

Responsibilities:
1. Explain algorithms and provide conceptual instruction.
2. Manage beginner guidance and construct roadmaps.
3. Organize revision timelines and study plans.
4. Recommend structural learning resources.

Enterprise Constraints:
- Strict boundary: Must NEVER analyze raw profiles or recommend specific contest problems.
- Must execute deterministically, returning strongly-typed structured output schema.
- Uses Groq for high-throughput, low-latency instruction generation.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from state.cpstate import CPCoachState


# =====================================================
# OUTPUT SCHEMA
# =====================================================

class RevisionTask(BaseModel):
    topic: str = Field(description="The specific algorithmic topic to revise")
    timeframe: str = Field(description="Suggested timeframe (e.g., 'Day 1', 'Next Week')")
    focus_area: str = Field(description="Specific sub-concept or weakness to focus on")

class TeachingOutput(BaseModel):
    learning_content: str = Field(
        description="Comprehensive, beginner-friendly explanation of the requested algorithm or concept. Include 'What it is', 'Why it is needed', 'Intuition', and 'When to use it'."
    )
    resources: List[str] = Field(
        description="List of top-tier resources (e.g., specific YouTube channel names, GeeksforGeeks, CP-Algorithms)."
    )
    study_plan: List[str] = Field(
        description="Short, actionable roadmap or study plan based on user weaknesses."
    )
    revision_timeline: List[RevisionTask] = Field(
        description="Structured timeline of topics the user needs to revisit to resolve their weaknesses."
    )


# =====================================================
# LLM INSTANTIATION
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2, # Low temperature for deterministic instructional output
    max_retries=3
)


# =====================================================
# SYSTEM PROMPT
# =====================================================

TEACHING_AGENT_PROMPT = """You are the **Teaching Agent** inside the CP Coach Multi-Agent Architecture.

**Role & Objective:**
You specialize in breaking down complex Data Structures and Algorithms (DSA) into intuitive, beginner-friendly narratives. You design personalized learning roadmaps, recommend high-yield external resources, and formulate spaced-repetition revision timelines for competitive programmers.

**Core Directives:**
1. **Explain Concepts:** Break down algorithms exactly into 4 parts: What it is, Why it is needed, Core Intuition, and When to apply it.
2. **Beginner Guidance:** Provide structured learning steps (Language syntax -> Basic Math -> Arrays -> etc) if the user is a novice.
3. **Revision Timelines:** Dynamically map the user's supplied 'weaknesses' into an actionable spaced `revision_timeline`.
4. **Hard Boundaries:** DO NOT analyze platform profiles. DO NOT recommend specific LeetCode/Codeforces problems (that is the Recommender Agent's job).

**Tone:** Academic, encouraging, extremely clear, and completely free of confusing jargon unless explicitly defined.
"""

# =====================================================
# PROMPT COMPOSITION
# =====================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", TEACHING_AGENT_PROMPT),
        (
            "human",
            """
            **Execution Context:**
            User Query: {query}
            Current Strengths: {strengths}
            Current Weaknesses: {weaknesses}
            
            Synthesize your instruction and timeline strictly matching the required schema.
            """
        )
    ]
)


# =====================================================
# CHAIN DEFINITION
# =====================================================

teaching_chain = prompt | llm.with_structured_output(TeachingOutput)


# =====================================================
# LANGGRAPH NODE EXECUTOR
# =====================================================

def teaching_agent_node(state: CPCoachState) -> Dict[str, Any]:
    """
    Executes the Teaching workflow within the CP Coach LangGraph.
    
    Reads: user_query, strengths, weaknesses
    Writes: learning_content, resources, study_plan, revision_timeline
    """
    
    # Safe extraction of state inputs
    query = state.get("user_query", "Give me a general beginner roadmap for CP.")
    strengths = state.get("strengths") or ["None identified"]
    weaknesses = state.get("weaknesses") or ["Fundamentals"]
    
    # Execute structured generation
    result: TeachingOutput = teaching_chain.invoke(
        {
            "query": query,
            "strengths": strengths,
            "weaknesses": weaknesses
        }
    )

    # Transform Pydantic objects back to state-compatible dictionaries
    revision_timeline_dict = [
        {"topic": task.topic, "timeframe": task.timeframe, "focus_area": task.focus_area}
        for task in result.revision_timeline
    ]

    # Return partial state updates (appended/merged dynamically by LangGraph reducer)
    return {
        "learning_content": result.learning_content,
        "resources": result.resources,
        "study_plan": result.study_plan,
        "revision_timeline": revision_timeline_dict
    }