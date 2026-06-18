import requests

def test_lc():
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
        "limit": 5,
        "filters": {
            "tags": ["dynamic-programming", "memoization"],
            "difficulty": "MEDIUM"
        }
    }

    url = "https://leetcode.com/graphql"
    response = requests.post(url, json={'query': query, 'variables': variables})
    print(response.json())

if __name__ == "__main__":
    test_lc()