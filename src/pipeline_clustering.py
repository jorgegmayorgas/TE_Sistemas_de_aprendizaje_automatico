import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# =========================================================
# 1. CONFIGURACIÓN
# =========================================================
INPUT_FILE = "data/dataset_recomendador_cursos.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# =========================================================
# 2. CARGA DE DATOS
# =========================================================
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("Primeras filas del dataset:")
print(df.head())
print("=" * 60)

print("\nInformación general:")
print(df.info())

print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nEstadísticos descriptivos:")
print(df.describe(include="all"))

# =========================================================
# 3. DEFINICIÓN DE VARIABLES
# =========================================================
target_id = "user_id"

numeric_features = [
    "age",
    "courses_viewed",
    "courses_completed",
    "avg_progress",
    "avg_rating_given",
    "total_watch_time_hours",
    "sessions_per_week",
    "avg_session_duration_min",
    "last_activity_days_ago",
    "completion_rate",
    "dropout_rate",
    "revisit_rate",
    "diversity_score",
    "engagement_score",
]

categorical_features = [
    "gender",
    "country",
    "education_level",
    "employment_status",
    "preferred_category",
    "skill_level",
    "learning_goal",
    "preferred_course_length",
    "preferred_content_type",
    "time_of_day_preference",
]

X = df[numeric_features + categorical_features].copy()

# =========================================================
# 4. PREPROCESAMIENTO
# =========================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_processed = preprocessor.fit_transform(X)

print("\nForma de X original:", X.shape)
print("Forma de X transformado:", X_processed.shape)

# =========================================================
# 5. ELECCIÓN DEL NÚMERO DE CLUSTERS
# =========================================================
inertias = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_processed)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_processed, labels)
    silhouette_scores.append(sil_score)
    print(f"k={k} | inertia={kmeans.inertia_:.2f} | silhouette={sil_score:.4f}")

# Graficar método del codo
plt.figure(figsize=(10, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Método del codo")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elbow_method.png"))
plt.close()

# Graficar silhouette
plt.figure(figsize=(10, 5))
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.title("Silhouette Score por número de clusters")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "silhouette_scores.png"))
plt.close()

# Elegimos el mejor k por silhouette
best_k = list(k_values)[np.argmax(silhouette_scores)]
print(f"\nMejor número de clusters según silhouette score: {best_k}")

# =========================================================
# 6. MODELO FINAL DE CLUSTERING
# =========================================================
final_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df["cluster"] = final_kmeans.fit_predict(X_processed)

print("\nDistribución de clusters:")
print(df["cluster"].value_counts().sort_index())

# =========================================================
# 7. REDUCCIÓN DE DIMENSIONALIDAD PARA VISUALIZACIÓN
# =========================================================
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed)

df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]

plt.figure(figsize=(10, 7))
for cluster_id in sorted(df["cluster"].unique()):
    subset = df[df["cluster"] == cluster_id]
    plt.scatter(
        subset["pca_1"],
        subset["pca_2"],
        label=f"Cluster {cluster_id}",
        alpha=0.6,
        s=25
    )

plt.title("Visualización de clusters con PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_visualization.png"))
plt.close()

# =========================================================
# 8. PERFILADO DE CLUSTERS
# =========================================================
numeric_profile = df.groupby("cluster")[numeric_features].mean().round(2)

categorical_profile = {}
for col in categorical_features:
    mode_per_cluster = df.groupby("cluster")[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    categorical_profile[col] = mode_per_cluster

categorical_profile_df = pd.DataFrame(categorical_profile)

cluster_profile = pd.concat([numeric_profile, categorical_profile_df], axis=1)

print("\nPerfil de clusters:")
print(cluster_profile)

cluster_profile.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles.csv"))

# =========================================================
# 9. FUNCIÓN SIMPLE DE ETIQUETADO DE CLUSTERS
# =========================================================
def assign_cluster_label(row):
    if row["completion_rate"] > 0.7 and row["engagement_score"] > 0.7:
        return "Usuarios muy comprometidos"
    elif row["completion_rate"] < 0.3 and row["last_activity_days_ago"] > 30:
        return "Usuarios con riesgo de abandono"
    elif row["diversity_score"] > 0.7 and row["courses_viewed"] > 20:
        return "Exploradores de contenido"
    elif row["skill_level"] == "Advanced":
        return "Usuarios avanzados"
    else:
        return "Usuarios intermedios/generalistas"

cluster_profile["cluster_label"] = cluster_profile.apply(assign_cluster_label, axis=1)

print("\nEtiquetas interpretativas de clusters:")
print(cluster_profile[["cluster_label"]])

cluster_profile.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles_labeled.csv"))

# =========================================================
# 10. GENERAR RECOMENDACIONES BÁSICAS POR CLUSTER
# =========================================================
def recommend_courses(cluster_row):
    category = cluster_row["preferred_category"]
    level = cluster_row["skill_level"]
    goal = cluster_row["learning_goal"]
    label = cluster_row["cluster_label"]

    if label == "Usuarios muy comprometidos":
        return f"Recomendar rutas avanzadas de {category}, especializaciones largas y proyectos prácticos de nivel {level}."
    elif label == "Usuarios con riesgo de abandono":
        return f"Recomendar cursos cortos de {category}, contenido guiado, recordatorios y rutas introductorias orientadas a {goal}."
    elif label == "Exploradores de contenido":
        return f"Recomendar packs variados de {category}, microcursos y contenidos relacionados de distintas subáreas."
    elif label == "Usuarios avanzados":
        return f"Recomendar cursos expertos, laboratorios y casos reales en {category}."
    else:
        return f"Recomendar itinerarios progresivos de {category} para nivel {level}, alineados con el objetivo {goal}."

cluster_profile["recommendation_strategy"] = cluster_profile.apply(recommend_courses, axis=1)

print("\nEstrategia de recomendación por cluster:")
print(cluster_profile[["cluster_label", "recommendation_strategy"]])

cluster_profile.to_csv(os.path.join(OUTPUT_DIR, "cluster_recommendation_strategy.csv"))

# =========================================================
# 11. EXPORTAR DATASET FINAL CON CLUSTERS
# =========================================================
df.to_csv(os.path.join(OUTPUT_DIR, "dataset_con_clusters.csv"), index=False)

print("\nProceso completado correctamente.")
print(f"Archivos generados en: {OUTPUT_DIR}")