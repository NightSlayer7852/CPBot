"""
recommendation_agent.py

Responsibilities:
1. Parse weaknesses, solved history, and topic mastery state (UFP).
2. Formulate highly granular LeetCode fetching parameters.
3. Recommend EXACT real problems by triggering API tool logic.
4. Scale difficulty dynamically (Easy -> Medium -> Hard) based on historical state.

It NEVER:
- Teaches the algorithms/concepts directly.
- Recommends already 'solved_problems'.
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
    topic_slugs: List[str] = Field(
        description="A list of lowercase hyphenated LeetCode topic slugs to fetch intersection tags. For subcategories, combine standard tags like ['dynamic-programming', 'memoization'] or ['dynamic-programming', 'bitmask']."
    )
    difficulty: str = Field(
        description="Target difficulty scaling dynamically based on user's topic_mastery. Exactly matching: 'EASY', 'MEDIUM', or 'HARD'."
    )
    count: int = Field(description="Number of problems to fetch for. (1-3)")

class RecommendationOutput(BaseModel):
    focus_topics: List[str] = Field(description="Granular topics or subtopics the user needs to practice next.")
    search_queries: List[SearchQuery] = Field(description="Instructions for the agent to fetch exact problems.")
    reasoning: str = Field(description="Detailed personalized explanation of why these exact subcategories and difficulty levels were chosen based on their track record (UFP).")


# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2, # Slightly bumped to allow creative but deterministic tag intersections
    max_retries=3
)


# =====================================================
# SYSTEM PROMPT
# =====================================================

RECOMMENDATION_PROMPT = """
You are the Deep Personalization Recommendation Agent of CP Coach.

Your specific job:
1. deeply analyze the user's User Profile Data (UFP), including 'strengths', 'weaknesses', and 'topic_mastery'.
2. NEVER give a generic response. Analyze their specific solved counts to scale difficulty dynamically. If they consistently solve Easy problems in a topic, graduate them to Medium. If they are failing Medium, drop back to Easy or suggest a specific Subcategory payload.
3. Extract precise subcategories. Instead of generic "Dynamic Programming", use intersecting tag requests like `['dynamic-programming', 'memoization']`, `['dynamic-programming', 'tree']`, or `['graph', 'depth-first-search']`.
4. Create structural Search Queries leveraging these subcategories.

Rules:
1. Base the `difficulty` directly on their `topic_mastery` track record. Do not blind-guess.
2. Max 5 total problems combined across queries.
3. Return only standard structural parameters. The system will auto-filter out `solved_problems` at the API layer.
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
            Topic Mastery State: {topic_mastery}
            
            Synthesize the optimal progression sequence. Output your structural search and deep, personalized reasoning.
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
    strengths = state.get("strengths", ["None"])
    weaknesses = state.get("weaknesses", ["None"])
    profile_summary = state.get("profile_summary", "New user.")
    topic_mastery = state.get("topic_mastery", {})
    solved_problems = state.get("solved_problems", [])

    # 1. Ask LLM for what to search
    result: RecommendationOutput = recommendation_chain.invoke(
        {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "profile_summary": profile_summary,
            "topic_mastery": topic_mastery
        }
    )

    # 2. Fetch EXACT problems from LeetCode GraphQL
    fetched_problems = []
    
    for query in result.search_queries:
        problems = fetch_leetcode_problems(
            topic_slugs=query.topic_slugs, 
            difficulty=query.difficulty, 
            limit=query.count,
            exclude_slugs=solved_problems
        )
        fetched_problems.extend(problems)

    return {
        "focus_topics": result.focus_topics,
        "recommended_problems": fetched_problems,
        "recommendation_reasoning": result.reasoning
    }