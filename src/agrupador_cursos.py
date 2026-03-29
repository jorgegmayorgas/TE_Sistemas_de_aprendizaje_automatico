import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# =========================================================
# 1. CONFIGURACIÓN
# =========================================================
INPUT_FILE = "data/dataset_recomendador_cursos.csv"


# =========================================================
# 2. CARGA DE DATOS
# =========================================================
df = pd.read_csv(INPUT_FILE)
print(df.groupby('preferred_category').count()['user_id'])
print("")
INPUT_FILE = "data/courses_catalog.csv"
df_courses = pd.read_csv(INPUT_FILE)
print(df_courses.groupby('category').count())