import pandas as pd
import numpy as np
import random

np.random.seed(42)

N = 500

categories = {
    "AI": ["Deep Learning", "NLP", "Computer Vision"],
    "Data Science": ["Pandas", "Statistics", "Visualization"],
    "Web Dev": ["Frontend", "Backend", "Fullstack"],
    "Marketing": ["SEO", "Ads", "Content"],
    "Cybersecurity": ["Ethical Hacking", "Network Security", "Forensics"]
}

difficulty_levels = ["Beginner", "Intermediate", "Advanced"]
content_types = ["Video", "Text", "Mixed"]
languages = ["Spanish", "English"]

data = []

for i in range(N):
    category = random.choice(list(categories.keys()))
    sub_category = random.choice(categories[category])

    difficulty = random.choice(difficulty_levels)
    duration = np.round(np.random.uniform(1, 50), 2)

    rating = np.round(np.random.uniform(3.0, 5.0), 2)

    data.append([
        i,
        f"{sub_category} Course {i}",
        category,
        sub_category,
        difficulty,
        duration,
        random.choice(content_types),
        rating,
        np.random.randint(10, 5000),
        random.choice([True, False]),
        random.choice(languages),
        f"{category},{sub_category},{difficulty}"
    ])

columns = [
    "course_id","title","category","sub_category","difficulty_level",
    "duration_hours","content_type","rating_avg","num_reviews",
    "is_certified","language","tags"
]

df_courses = pd.DataFrame(data, columns=columns)
df_courses.to_csv("data/courses_catalog.csv", index=False)

print("Dataset de cursos generado")