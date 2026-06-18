import requests
from typing import List, Dict

def fetch_leetcode_problems(topic_slug: str, difficulty: str = "MEDIUM", limit: int = 5) -> List[Dict]:
    """
    Fetches exact problems from LeetCode GraphQL API.
    
    Args:
        topic_slug: The hyphenated topic name (e.g., 'dynamic-programming', 'arrays')
        difficulty: 'EASY', 'MEDIUM', or 'HARD'
        limit: Number of problems to fetch
    """
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
        "limit": limit,
        "filters": {
            "tags": [topic_slug],
            "difficulty": difficulty.upper()
        }
    }

    url = "https://leetcode.com/graphql"
    
    try:
        response = requests.post(url, json={'query': query, 'variables': variables}, timeout=10)
        response.raise_for_status()
        data = response.json()
        questions = data.get('data', {}).get('problemsetQuestionList', {}).get('questions', [])
        
        return [
            {
                "platform": "LeetCode",
                "title": q["title"],
                "url": f"https://leetcode.com/problems/{q['titleSlug']}/",
                "difficulty": q["difficulty"].capitalize()
            }
            for q in questions
        ]
    except Exception as e:
        print(f"LeetCode fetch error: {e}")
        return []
