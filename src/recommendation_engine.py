import pandas as pd

def recommend_courses(user_row, courses_df, top_n=5):

    category = user_row["preferred_category"]
    level = user_row["skill_level"]
    content_type = user_row["preferred_content_type"]

    # Filtrado inicial
    filtered = courses_df[
        (courses_df["category"] == category) &
        (courses_df["difficulty_level"] == level)
    ]

    # Score personalizado
    filtered = filtered.copy()

    filtered["score"] = (
        filtered["rating_avg"] * 0.5 +
        filtered["num_reviews"] * 0.0001 +
        (filtered["content_type"] == content_type).astype(int) * 0.3 +
        filtered["is_certified"].astype(int) * 0.2
    )

    return filtered.sort_values(by="score", ascending=False).head(top_n)