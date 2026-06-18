"""
recommendation_agent.py

Responsibilities:
1. Parse weaknesses to identify focus topics.
2. Formulate LeetCode fetching parameters.
3. Recommend EXACT real problems by triggering API tool logic.
4. Explain WHY those topics and problems were selected.

It NEVER:
- Teaches the algorithms/concepts directly.
- Modifies profile directly without graph confirmation.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from state.cpstate import CPCoachState
from tools.recommender.new_problem import fetch_leetcode_problems


# =====================================================
# OUTPUT SCHEMA
# =====================================================

class SearchQuery(BaseModel):
    topic_slug: str = Field(
        description="A valid lowercase hyphenated LeetCode topic slug (e.g., 'dynamic-programming', 'two-pointers', 'array', 'binary-search')."
    )
    difficulty: str = Field(
        description="Target difficulty level exactly matching: 'EASY', 'MEDIUM', or 'HARD'."
    )
    count: int = Field(description="Number of problems to fetch for this topic (1-3).")

class RecommendationOutput(BaseModel):
    focus_topics: List[str] = Field(description="Top concepts the user needs to practice next.")
    search_queries: List[SearchQuery] = Field(description="Instructions for the agent to fetch exact problems.")
    reasoning: str = Field(description="A brief explanation logically justifying why these topics were chosen.")


# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_retries=3
)


# =====================================================
# SYSTEM PROMPT
# =====================================================

RECOMMENDATION_PROMPT = """
You are the Recommendation Agent of CP Coach.

Your specific job:
1. Look at the user's weaknesses and profile summary.
2. Select the most urgent topics to practice right now.
3. Generate exact structural queries to fetch LeetCode problems.
4. Briefly justify your reasoning.

Rules:
1. Target progression over immediate mastery. For a beginner struggling with standard concepts, request 'EASY' or 'MEDIUM' difficulties.
2. Map their weaknesses to standard LeetCode tags (e.g. 'hash-table', 'depth-first-search', 'math', 'greedy').
3. Keep the total problem count requested across all queries to 5 or fewer.
4. Never hallucinatively recommend problems by name—ONLY output standard `search_queries` so the Live API can securely fetch them.
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
            Strengths: {strengths}
            Weaknesses: {weaknesses}
            Profile Summary: {profile_summary}
            
            Based on the above, which tags/difficulties should we fetch problems for?
            """
        )
    ]
)

# =====================================================
# CHAIN
# =====================================================

recommendation_chain = prompt | llm.with_structured_output(RecommendationOutput)


# =====================================================
# NODE
# =====================================================

def recommendation_agent_node(state: CPCoachState) -> Dict[str, Any]:
    """
    Executes the LLM request to find focus topics, runs the Leetcode Tool manually 
    with the generated queries, and appends EXACT problems to the state graph.
    """
    strengths = state.get("strengths", ["None"])
    weaknesses = state.get("weaknesses", ["None"])
    profile_summary = state.get("profile_summary", "New user. No data.")

    # 1. Ask LLM for what to search
    result: RecommendationOutput = recommendation_chain.invoke(
        {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "profile_summary": profile_summary
        }
    )

    # 2. Fetch EXACT problems from LeetCode GraphQL using the requested parameters
    fetched_problems = []
    
    for query in result.search_queries:
        problems = fetch_leetcode_problems(
            topic_slug=query.topic_slug, 
            difficulty=query.difficulty, 
            limit=query.count
        )
        fetched_problems.extend(problems)

    return {
        "focus_topics": result.focus_topics,
        "recommended_problems": fetched_problems,
        "recommendation_reasoning": result.reasoning
    }