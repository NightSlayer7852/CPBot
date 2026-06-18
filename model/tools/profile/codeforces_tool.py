import requests


BASE_URL = "https://codeforces.com/api"


def get_codeforces_profile(handle: str):
    """
    Returns:
    {
        rating,
        max_rating,
        rank,
        tag_distribution
    }
    """

    # User Info
    user_info = requests.get(
        f"{BASE_URL}/user.info",
        params={
            "handles": handle
        }
    ).json()

    user = user_info["result"][0]

    # Recent submissions
    submissions = requests.get(
        f"{BASE_URL}/user.status",
        params={
            "handle": handle,
            "from": 1,
            "count": 500
        }
    ).json()

    tag_distribution = {}

    for submission in submissions["result"]:

        if submission["verdict"] != "OK":
            continue

        tags = submission["problem"].get(
            "tags",
            []
        )

        for tag in tags:
            tag_distribution[tag] = (
                tag_distribution.get(tag, 0) + 1
            )

    return {
        "rating": user.get("rating", 0),
        "max_rating": user.get("maxRating", 0),
        "rank": user.get("rank", "unrated"),
        "tag_distribution": tag_distribution
    }