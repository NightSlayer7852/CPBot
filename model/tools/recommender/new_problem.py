import requests
from typing import List, Dict

def fetch_leetcode_problems(topic_slugs: List[str], difficulty: str = "MEDIUM", limit: int = 5, exclude_slugs: List[str] = None) -> List[Dict]:
    """
    Fetches exact problems from LeetCode GraphQL API.
    
    Args:
        topic_slugs: List of hyphenated topic names (e.g., ['dynamic-programming', 'memoization']) to intersect tags.
        difficulty: 'EASY', 'MEDIUM', or 'HARD'
        limit: Number of problems to fetch
        exclude_slugs: List of 'titleSlug' strings of problems the user has already solved.
    """
    if exclude_slugs is None:
        exclude_slugs = []
        
    query = """
    query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
      problemsetQuestionList: questionList(
        categorySlug: $categorySlug
        limit: $limit
        skip: $skip
        filters: $filters
      ) {
        questions: data {
          difficulty
          title
          titleSlug
          topicTags {
            name
          }
        }
      }
    }
    """

    variables = {
        "categorySlug": "",
        "skip": 0,
        "limit": limit + len(exclude_slugs), # Over-fetch in case we filter out solved problems
        "filters": {
            "tags": topic_slugs,
            "difficulty": difficulty.upper()
        }
    }

    url = "https://leetcode.com/graphql"
    
    try:
        response = requests.post(url, json={'query': query, 'variables': variables}, timeout=10)
        response.raise_for_status()
        data = response.json()
        questions = data.get('data', {}).get('problemsetQuestionList', {}).get('questions', [])
        
        # Filter out previously solved problems
        unsolved = [q for q in questions if q['titleSlug'] not in exclude_slugs]
        
        return [
            {
                "platform": "LeetCode",
                "title": q["title"],
                "url": f"https://leetcode.com/problems/{q['titleSlug']}/",
                "difficulty": q["difficulty"].capitalize()
            }
            for q in unsolved[:limit]
        ]
    except Exception as e:
        print(f"LeetCode fetch error: {e}")
        return []
