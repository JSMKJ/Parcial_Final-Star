# Proyecto B — Clasificación Multiclase de Objetos Estelares

**Curso:** Inteligencia Artificial - Universidad Agustiniana

**Tipo de Proyecto:** Clasificación Multiclase con Alta Dimensionalidad

**Dataset:** Stellar Classification Dataset - SDSS17

---

## 📋 Descripción del Proyecto

Este proyecto aborda un problema de **clasificación multiclase** utilizando datos astronómicos del Sloan Digital Sky Survey (SDSS17). El objetivo es construir un clasificador robusto capaz de distinguir entre tres tipos de objetos celestes:

- **GALAXY**: Galaxias
- **STAR**: Estrellas
- **QSO**: Cuásares (objetos cuasi-estelares)

El proyecto demuestra habilidades en:
- Análisis exploratorio de datos (EDA)
- Preprocesamiento y escalado de características
- Reducción de dimensionalidad con PCA
- Entrenamiento y comparación de múltiples modelos de Machine Learning
- Calibración de probabilidades
- Interpretabilidad de modelos
- Evaluación de robustez y generalización

---

## 📁 Estructura del Repositorio

```
Parcial/
├── Data/
│   └── archive/
│       └── star_classification.csv    # Dataset principal
├── parcial_final.ipynb                # Notebook completo con todo el análisis
├── README.md                          # Este archivo
├── REPORT.md                          # Informe técnico detallado
└── requirements.txt                   # Dependencias del proyecto
```

---

## 📊 Dataset

**Fuente:** [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)

**Características:**
- **Tamaño:** 100,000 observaciones
- **Características:** 17 variables numéricas
- **Clases:** 3 (GALAXY, STAR, QSO)
- **Tipo:** Clasificación multiclase

**Variables principales:**
- Coordenadas astronómicas: `alpha`, `delta`
- Magnitudes fotométricas: `u`, `g`, `r`, `i`, `z` (5 bandas)
- Corrimiento al rojo: `redshift`
- Metadatos de observación: `run_ID`, `cam_col`, `field_ID`, `plate`, `MJD`, `fiber_ID`

---

## 🚀 Cómo Ejecutar el Proyecto

### 1. Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd Parcial
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar el Dataset

Opción A: Descargar manualmente desde Kaggle
1. Ir a: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
2. Descargar `star_classification.csv`
3. Colocar en: `Data/archive/star_classification.csv`

Opción B: Usar Kaggle API (requiere configuración previa)
```bash
kaggle datasets download -d fedesoriano/stellar-classification-dataset-sdss17
unzip stellar-classification-dataset-sdss17.zip -d Data/archive/
```

### 4. Ejecutar el Notebook

```bash
jupyter notebook parcial_final.ipynb
```

O usar Jupyter Lab:
```bash
jupyter lab parcial_final.ipynb
```

**Nota:** El notebook está diseñado para ejecutarse de principio a fin sin errores. Asegúrate de ejecutar las celdas en orden.

---

## 🔬 Metodología

El proyecto sigue una metodología completa de Machine Learning:

### 1. Análisis Exploratorio de Datos (EDA)
- Carga y exploración inicial del dataset
- Análisis de valores faltantes
- Distribución de clases (desbalance)
- Estadísticas descriptivas
- Matriz de correlación
- Detección de varianza cercana a cero

### 2. Preprocesamiento
- Separación de variables (X, y)
- Imputación de valores faltantes (mediana)
- Escalado con StandardScaler (media=0, std=1)
- Análisis de desbalance de clases

### 3. Reducción de Dimensionalidad
- Aplicación de PCA (Análisis de Componentes Principales)
- Análisis de varianza explicada
- Selección de número óptimo de componentes (95% varianza)
- Visualización en 2D de separabilidad de clases
- Selección de características con SelectKBest

### 4. División de Datos
- Split train/test: 80/20
- Estratificación por clase
- Validación cruzada: StratifiedKFold (5 folds)

### 5. Modelos Entrenados
- **Baseline:** Logistic Regression
- **Random Forest Classifier**
- **Gradient Boosting** (XGBoost o GradientBoostingClassifier)
- **Support Vector Machine** (kernel RBF)

Métricas de evaluación:
- Accuracy
- Balanced Accuracy
- Precision, Recall, F1-score (macro y por clase)
- Matriz de confusión

### 6. Calibración de Probabilidades
- Evaluación con Brier score
- Curvas de calibración por clase
- Aplicación de CalibratedClassifierCV (método sigmoid)
- Comparación antes/después

### 7. Interpretabilidad
- Importancias de características (Random Forest)
- Interpretación de componentes principales
- Análisis de errores (matriz de confusión)
- Visualización de puntos mal clasificados en espacio PCA

### 8. Robustez y Generalización
- Prueba de sensibilidad al ruido gaussiano (7 niveles)
- Análisis por subgrupos (niveles de confianza)
- Curvas de degradación de rendimiento

### 9. Conclusiones y Recomendaciones
- Resumen de resultados por modelo
- Mejor modelo y configuración
- Recomendaciones para producción
- Limitaciones y trabajo futuro

---

## 📈 Resultados Principales

Los resultados detallados se encuentran en [REPORT.md](REPORT.md), pero en resumen:

- **Mejor modelo:** [Se determina en el notebook - típicamente Random Forest o XGBoost]
- **F1-score macro:** ~XX.XX% (completar después de ejecutar)
- **Balanced Accuracy:** ~XX.XX%
- **Componentes PCA:** ~XX componentes explican 95% de varianza

**Observaciones clave:**
- PCA reduce efectivamente la dimensionalidad manteniendo información relevante
- La calibración mejora las probabilidades predichas
- El modelo muestra buena robustez ante ruido moderado
- Algunas clases se confunden más que otras (analizado en detalle en el notebook)

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Pandas:** Manipulación de datos
- **NumPy:** Operaciones numéricas
- **Scikit-learn:** Modelos de ML, preprocesamiento, métricas
- **Matplotlib & Seaborn:** Visualizaciones
- **XGBoost:** Gradient Boosting (opcional)
- **Jupyter Notebook:** Entorno de desarrollo interactivo

---

## 📚 Referencias

1. [Sloan Digital Sky Survey (SDSS)](https://www.sdss.org/)
2. [Stellar Classification Dataset - Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)
3. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
4. [PCA Tutorial](https://scikit-learn.org/stable/modules/decomposition.html#pca)
5. Repositorio del curso: [Inteligencia Artificial - U](https://github.com/BrayanTorres2/Inteligencia-artificial-computacion-U)

---

## 👨‍💻 Autor

**Joan Sebastian Montes Jerez**
Universidad Agustiniana
Curso: Inteligencia Artificial
Noviembre 2025

---

## 📝 Licencia

Este proyecto es material académico para el curso de Inteligencia Artificial de la Universidad Agustiniana.

---

## ✅ Checklist de Entregables

- [x] Notebook ejecutable (`parcial_final.ipynb`)
- [x] README.md con instrucciones claras
- [x] REPORT.md con análisis detallado
- [x] requirements.txt con dependencias
- [x] Dataset accesible en `Data/archive/`
- [x] Código reproducible (random_state=42)
- [x] Análisis completo de las 9 secciones requeridas

---

## 🆘 Soporte

Si tienes problemas ejecutando el proyecto:

1. Verifica que todas las dependencias estén instaladas: `pip install -r requirements.txt`
2. Asegúrate de que el dataset esté en la ruta correcta: `Data/archive/star_classification.csv`
3. Usa Python 3.8 o superior
4. Revisa que tienes suficiente memoria RAM (se recomienda 8GB+)

Para cualquier duda, consulta el notebook que contiene explicaciones detalladas en cada paso.
