import os
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from src.recommendation_engine import recommend_courses

# =========================================================
# CONFIGURACIÓN
# =========================================================
app = FastAPI()

# Ruta base del proyecto (importante en Windows)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Templates (HTML)
templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "app", "templates")
)

# =========================================================
# CARGA DE DATOS
# =========================================================
try:
    df_users = pd.read_csv(
        os.path.join(BASE_DIR, "outputs", "dataset_con_clusters.csv")
    )
    df_courses = pd.read_csv(
        os.path.join(BASE_DIR, "data", "courses_catalog.csv")
    )

    print(f"Usuarios cargados: {len(df_users)}")
    print(f"Cursos cargados: {len(df_courses)}")

except Exception as e:
    print("ERROR cargando datos:", e)
    df_users = pd.DataFrame()
    df_courses = pd.DataFrame()
#preprocesamiento para el modelo de recomendación
# =========================================================
# PREPARAR MODELO DE SIMILITUD
# =========================================================

# Mapear categóricas a números simples
skill_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
category_map = {cat: i for i, cat in enumerate(df_users["preferred_category"].unique())}

df_users["skill_encoded"] = df_users["skill_level"].map(skill_map)
df_users["category_encoded"] = df_users["preferred_category"].map(category_map)

# Features para similitud
feature_cols = ["age", "engagement_score", "skill_encoded", "category_encoded"]

X_users = df_users[feature_cols].fillna(0)

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_users)

# Modelo vecinos
nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn_model.fit(X_scaled)
# =========================================================
# RUTA PRINCIPAL (HTML)
# =========================================================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

# =========================================================
# ENDPOINT DE RECOMENDACIÓN
# =========================================================
@app.get("/recommend")
def recommend(age: int, country: str):

    if df_users.empty or df_courses.empty:
        return {"error": "Datasets no cargados"}

    # Filtrar por país
    filtered = df_users[df_users["country"].str.lower() == country.lower()]

    if filtered.empty:
        return {"error": f"No hay usuarios en {country}"}

    # Buscar usuario más cercano en edad
    filtered["age_diff"] = (filtered["age"] - age).abs()
    user = filtered.sort_values("age_diff").iloc[0]

    try:
        recs = recommend_courses(user, df_courses)

        return {
            "user_used": {
                "user_id": int(user["user_id"]),
                "age": int(user["age"]),
                "country": user["country"],
                "cluster": int(user["cluster"])
            },
            "recommendations": recs.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}
# =========================================================
# ENDPOINT DE RECOMENDACIÓN INTELIGENTE (KNN)
# =========================================================
@app.get("/recommend-smart")
def recommend_smart(
    age: int,
    skill_level: str,
    category: str,
    engagement: float
):

    if df_users.empty:
        return {"error": "Dataset no cargado"}

    try:
        # Codificar inputs
        skill_encoded = skill_map.get(skill_level)
        category_encoded = category_map.get(category)

        if skill_encoded is None or category_encoded is None:
            return {"error": "Skill level o categoría inválida"}

        # Crear vector de entrada
        input_vector = [[age, engagement, skill_encoded, category_encoded]]

        input_scaled = scaler.transform(input_vector)

        # Buscar vecino más cercano
        dist, idx = nn_model.kneighbors(input_scaled)

        similar_user = df_users.iloc[idx[0][0]]

        # Obtener recomendaciones
        recs = recommend_courses(similar_user, df_courses)

        return {
            "input_user": {
                "age": age,
                "skill_level": skill_level,
                "category": category,
                "engagement": engagement
            },
            "matched_user": {
                "user_id": int(similar_user["user_id"]),
                "cluster": int(similar_user["cluster"]),
                "age": int(similar_user["age"])
            },
            "recommendations": recs.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}
# =========================================================
# HEALTH CHECK (opcional pero útil)
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}