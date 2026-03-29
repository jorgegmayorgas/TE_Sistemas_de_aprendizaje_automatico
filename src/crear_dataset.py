import pandas as pd
import numpy as np
import random

np.random.seed(42)

N = 50000

# ------------------------
# CATEGORÍAS
# ------------------------
genders = ["Male", "Female", "Other"]
countries = ["Spain", "Mexico", "Argentina", "Colombia", "Chile"]
education_levels = ["Basic", "Intermediate", "University", "Master"]
employment_statuses = ["Student", "Employed", "Unemployed"]

categories = ["AI", "Data Science", "Web Dev", "Marketing", "Cybersecurity"]
skill_levels = ["Beginner", "Intermediate", "Advanced"]
learning_goals = ["Career Change", "Upskill", "Hobby"]

course_length = ["Short", "Medium", "Long"]
content_type = ["Video", "Text", "Mixed"]
time_pref = ["Morning", "Afternoon", "Night"]

# ------------------------
# GENERACIÓN
# ------------------------
data = []

for i in range(N):
    age = np.random.randint(18, 65)

    courses_viewed = np.random.randint(1, 50)
    courses_completed = np.random.randint(0, courses_viewed)

    avg_progress = np.round(np.random.uniform(10, 100), 2)
    avg_rating = np.round(np.random.uniform(2.5, 5), 2)

    watch_time = np.round(np.random.uniform(5, 300), 2)
    sessions_week = np.random.randint(1, 14)
    session_duration = np.round(np.random.uniform(5, 120), 2)

    last_activity = np.random.randint(0, 90)

    completion_rate = courses_completed / max(courses_viewed, 1)
    dropout_rate = 1 - completion_rate

    revisit_rate = np.round(np.random.uniform(0, 1), 2)
    diversity_score = np.round(np.random.uniform(0.1, 1), 2)

    # Engagement score (muy importante para clustering)
    engagement_score = (
        0.3 * completion_rate +
        0.2 * revisit_rate +
        0.2 * (avg_progress / 100) +
        0.3 * (sessions_week / 14)
    )

    data.append([
        i,
        age,
        random.choice(genders),
        random.choice(countries),
        random.choice(education_levels),
        random.choice(employment_statuses),
        random.choice(categories),
        random.choice(skill_levels),
        random.choice(learning_goals),
        courses_viewed,
        courses_completed,
        avg_progress,
        avg_rating,
        watch_time,
        sessions_week,
        session_duration,
        last_activity,
        completion_rate,
        dropout_rate,
        revisit_rate,
        diversity_score,
        engagement_score,
        random.choice(course_length),
        random.choice(content_type),
        random.choice(time_pref)
    ])

# ------------------------
# DATAFRAME
# ------------------------
columns = [
    "user_id", "age", "gender", "country", "education_level", "employment_status",
    "preferred_category", "skill_level", "learning_goal",
    "courses_viewed", "courses_completed", "avg_progress", "avg_rating_given",
    "total_watch_time_hours", "sessions_per_week", "avg_session_duration_min",
    "last_activity_days_ago",
    "completion_rate", "dropout_rate", "revisit_rate", "diversity_score", "engagement_score",
    "preferred_course_length", "preferred_content_type", "time_of_day_preference"
]

df = pd.DataFrame(data, columns=columns)

# ------------------------
# GUARDAR CSV
# ------------------------
df.to_csv("./data/dataset_recomendador_cursos.csv", index=False)

print("Dataset generado correctamente")
print(df.head())