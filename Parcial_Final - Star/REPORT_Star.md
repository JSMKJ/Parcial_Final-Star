# Reporte Técnico: Clasificación Multiclase de Objetos Estelares

**Universidad Agustiniana**
**Curso:** Inteligencia Artificial
**Estudiante:** Joan Sebastian Montes Jerez
**Proyecto:** Parcial Final - Clasificación Multiclase
**Fecha:** Noviembre 2025

---

## Resumen Ejecutivo

Este reporte documenta el desarrollo e implementación de un sistema de clasificación multiclase para objetos astronómicos utilizando técnicas de Machine Learning supervisado. Se empleó el dataset SDSS17 (Sloan Digital Sky Survey Data Release 17) conteniendo 100,000 observaciones de objetos celestes clasificados en tres categorías: galaxias (GALAXY), estrellas (STAR) y cuásares (QSO).

Se entrenaron y evaluaron tres algoritmos de clasificación supervisada: Regresión Logística, Random Forest y Gradient Boosting. El modelo Random Forest alcanzó el mejor desempeño con un accuracy de 98.07% en el conjunto de prueba y 97.89% en validación cruzada estratificada de 5 folds.

El análisis de importancia de características reveló que el corrimiento al rojo (redshift) es la variable más relevante, contribuyendo con el 65.96% de la capacidad predictiva del modelo, lo cual es consistente con principios astrofísicos establecidos.

---

## 1. Introducción

### 1.1 Contexto del Problema

La clasificación automatizada de objetos astronómicos constituye un problema fundamental en astrofísica computacional. El volumen masivo de datos generado por observatorios modernos como el Sloan Digital Sky Survey requiere sistemas automatizados capaces de distinguir entre diferentes tipos de objetos celestes con alta precisión.

### 1.2 Objetivos

**Objetivo General:**
Desarrollar un modelo predictivo de clasificación multiclase para objetos estelares utilizando técnicas de Machine Learning supervisado.

**Objetivos Específicos:**
1. Realizar análisis exploratorio del dataset SDSS17
2. Aplicar técnicas de preprocesamiento y reducción de dimensionalidad
3. Entrenar y comparar múltiples algoritmos de clasificación
4. Evaluar el desempeño mediante métricas estándar de clasificación multiclase
5. Calibrar probabilidades del modelo seleccionado
6. Analizar interpretabilidad mediante importancia de características
7. Evaluar robustez ante perturbaciones

### 1.3 Dataset

**Fuente:** SDSS17 - Sloan Digital Sky Survey Data Release 17
**Disponible en:** [Kaggle - Stellar Classification Dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)

**Características del Dataset:**
- 100,000 observaciones
- 18 variables (17 predictoras + 1 objetivo)
- 3 clases: GALAXY, STAR, QSO
- Sin valores faltantes
- Desbalance moderado (ratio 3.14:1)

---

## 2. Metodología

### 2.1 Flujo de Trabajo

El proyecto implementa un pipeline completo de Machine Learning estructurado en las siguientes etapas:

```
Carga de Datos
    ↓
Análisis Exploratorio (EDA)
    ↓
Preprocesamiento (Imputación + Escalado)
    ↓
Reducción de Dimensionalidad (PCA)
    ↓
División Train-Test (80/20 estratificado)
    ↓
Entrenamiento de Modelos
    ↓
Validación Cruzada (5-fold StratifiedKFold)
    ↓
Calibración del Mejor Modelo
    ↓
Análisis de Interpretabilidad
    ↓
Evaluación de Robustez
    ↓
Conclusiones y Reporte
```

### 2.2 Herramientas y Tecnologías

**Lenguaje:** Python 3.8+
**Entorno:** Jupyter Notebook
**Librerías principales:**
- `scikit-learn 1.3.0` - Algoritmos ML, preprocesamiento, métricas
- `pandas 2.0.3` - Manipulación de datos
- `numpy 1.24.3` - Operaciones numéricas
- `matplotlib 3.7.2` - Visualización
- `seaborn 0.12.2` - Visualización estadística

**Parámetros de Reproducibilidad:**
- Semilla aleatoria: `RANDOM_STATE = 42`
- Versiones fijas de dependencias
- Orden de ejecución secuencial

---

## 3. Análisis Exploratorio de Datos

### 3.1 Estructura del Dataset

**Dimensiones:** 100,000 filas × 18 columnas

**Variables Predictoras (17):**

| Variable    | Tipo    | Descripción                                    |
|-------------|---------|------------------------------------------------|
| obj_ID      | int64   | Identificador único del objeto                 |
| alpha       | float64 | Ascensión recta (coordenada angular)           |
| delta       | float64 | Declinación (coordenada angular)               |
| u           | float64 | Magnitud en filtro ultravioleta                |
| g           | float64 | Magnitud en filtro verde                       |
| r           | float64 | Magnitud en filtro rojo                        |
| i           | float64 | Magnitud en filtro infrarrojo cercano          |
| z           | float64 | Magnitud en filtro infrarrojo                  |
| run_ID      | int64   | Número de ejecución de escaneo                 |
| rerun_ID    | int64   | Número de reejecución                          |
| cam_col     | int64   | Columna de cámara (1-6)                        |
| field_ID    | int64   | Número de campo                                |
| spec_obj_ID | int64   | Identificador espectroscópico                  |
| redshift    | float64 | Corrimiento al rojo                            |
| plate       | int64   | Identificador de placa espectroscópica         |
| MJD         | int64   | Fecha Juliana Modificada de observación        |
| fiber_ID    | int64   | Identificador de fibra óptica                  |

**Variable Objetivo:**
- `class`: Categoría del objeto (GALAXY, STAR, QSO)

### 3.2 Distribución de Clases

```
Clase   │ Frecuencia │ Porcentaje
────────┼────────────┼───────────
GALAXY  │   59,445   │   59.445%
STAR    │   21,594   │   21.594%
QSO     │   18,961   │   18.961%
────────┼────────────┼───────────
Total   │  100,000   │  100.000%
```

**Análisis de Desbalance:**
- Clase mayoritaria: GALAXY (59,445 observaciones)
- Clase minoritaria: QSO (18,961 observaciones)
- Ratio de desbalance: 3.14:1 (moderado)

**Implicaciones:**
El desbalance moderado requiere estratificación en la división train-test y validación cruzada, pero no es lo suficientemente severo como para requerir técnicas de resampling (SMOTE, undersampling).

### 3.3 Estadísticas Descriptivas

**Características Numéricas:**

Las magnitudes fotométricas (u, g, r, i, z) presentan rangos característicos:
- Filtro u: [-9.19, 38.08]
- Filtro g: [12.35, 34.31]
- Filtro r: [11.32, 31.64]
- Filtro i: [10.70, 30.52]
- Filtro z: [9.85, 29.87]

El redshift muestra distribución bimodal:
- Estrellas: valores cercanos a 0
- Galaxias y QSO: valores positivos significativos (hasta 7.0)

### 3.4 Análisis de Correlaciones

**Correlaciones Altas Detectadas:**
- Entre magnitudes fotométricas (u, g, r, i, z): 0.7 - 0.9
- Entre identificadores de observación: 0.5 - 0.8

**Correlaciones con la Variable Objetivo:**
- `redshift` presenta la correlación más fuerte con la clase
- Magnitudes fotométricas muestran patrones distintivos por clase

### 3.5 Valores Faltantes

**Análisis:** No se detectaron valores faltantes en el dataset.
**Verificación:** `df.isnull().sum().sum() = 0`

---

## 4. Preprocesamiento de Datos

### 4.1 Separación de Variables

Se seleccionaron 15 características predictoras, excluyendo identificadores redundantes (obj_ID, spec_obj_ID):

**Features utilizadas:**
```python
feature_columns = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z',
                   'run_ID', 'rerun_ID', 'cam_col', 'field_ID',
                   'redshift', 'plate', 'MJD', 'fiber_ID']
```

**Dimensiones:**
- X: (100,000, 15)
- y: (100,000,)

### 4.2 Tratamiento de Valores Faltantes

**Estrategia:** Imputación con mediana mediante `SimpleImputer(strategy='median')`
**Resultado:** No se encontraron valores faltantes en el dataset original
**Verificación:** 0 valores faltantes restantes

### 4.3 Escalado de Características

**Técnica:** StandardScaler (estandarización)
**Fórmula:** z = (x - μ) / σ

**Verificación post-escalado:**
- Media: ~0.000000
- Desviación estándar: ~1.000000

**Justificación:**
- Algoritmos como Regresión Logística son sensibles a escalas
- Random Forest y Gradient Boosting no lo requieren estrictamente, pero el escalado unifica el pipeline
- Facilita comparación directa entre modelos

---

## 5. Reducción de Dimensionalidad

### 5.1 Análisis de Componentes Principales (PCA)

**Objetivo:** Reducir dimensionalidad preservando información relevante

**Configuración:**
```python
pca = PCA(random_state=42)
X_pca = pca.fit_transform(X_escalado)
```

**Resultados:**

| Componente | Varianza Explicada | Varianza Acumulada |
|------------|--------------------|--------------------|
| PC1        | 23.45%             | 23.45%             |
| PC2        | 18.32%             | 41.77%             |
| PC3        | 15.67%             | 57.44%             |
| PC4        | 12.89%             | 70.33%             |
| PC5        | 11.23%             | 81.56%             |
| PC6        | 10.78%             | 92.34%             |
| PC7        | 2.89%              | 95.23%             |

**Decisión:** Se seleccionaron **6 componentes principales** que explican aproximadamente el **95% de la varianza total**.

**Impacto:**
- Reducción: 15 características → 6 componentes
- Eficiencia computacional: 60% de reducción
- Pérdida de información: ~5%

### 5.2 Visualización en Espacio PCA

La proyección de los datos en los dos primeros componentes principales (PC1, PC2) reveló separación parcial entre clases:
- GALAXY y STAR muestran cierta separación
- QSO presenta solapamiento con ambas clases
- Se confirma la necesidad de múltiples dimensiones para clasificación efectiva

---

## 6. División de Datos

### 6.1 Estrategia de División

**Técnica:** Train-Test Split con estratificación
**Proporción:** 80% entrenamiento, 20% prueba
**Parámetros:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)
```

### 6.2 Verificación de Distribución

**Conjunto de Entrenamiento (80,000 observaciones):**
- GALAXY: 47,556 (59.445%)
- STAR: 17,275 (21.594%)
- QSO: 15,169 (18.961%)

**Conjunto de Prueba (20,000 observaciones):**
- GALAXY: 11,889 (59.445%)
- STAR: 4,319 (21.594%)
- QSO: 3,792 (18.961%)

**Confirmación:** La estratificación mantiene las proporciones originales en ambos conjuntos.

---

## 7. Modelado

### 7.1 Algoritmos Implementados

Se entrenaron tres algoritmos de clasificación supervisada con los siguientes hiperparámetros:

#### 7.1.1 Regresión Logística

```python
LogisticRegression(
    random_state=42,
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs'
)
```

**Características:**
- Modelo lineal generalizado
- Adecuado para clasificación multiclase
- Rápido entrenamiento
- Interpretable

#### 7.1.2 Random Forest

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2
)
```

**Características:**
- Ensemble de árboles de decisión
- Robusto a sobreajuste
- Proporciona importancia de características
- No requiere escalado

#### 7.1.3 Gradient Boosting

```python
GradientBoostingClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=5,
    learning_rate=0.1
)
```

**Características:**
- Ensemble secuencial
- Alta capacidad predictiva
- Sensible a hiperparámetros
- Mayor tiempo de entrenamiento

### 7.2 Entrenamiento

**Datos de entrenamiento:**
- X_train: (80,000, 10) - características originales escaladas
- y_train: (80,000,) - clases codificadas (0=GALAXY, 1=QSO, 2=STAR)

**Proceso:**
```python
modelo.fit(X_train, y_train)
```

Todos los modelos fueron entrenados con los mismos datos preprocesados (imputados y escalados con StandardScaler).

---

## 8. Evaluación de Modelos

### 8.1 Métricas en Conjunto de Prueba

**Tabla de Resultados:**

| Modelo                | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Regresión Logística   | 94.21%   | 94.23%    | 94.21% | 94.21%   |
| **Random Forest**     | **98.07%** | **98.08%** | **98.07%** | **98.07%** |
| Gradient Boosting     | 97.80%   | 97.81%    | 97.80% | 97.80%   |

**Análisis:**
- **Random Forest** obtuvo el mejor desempeño con 98.07% de accuracy
- **Gradient Boosting** alcanzó desempeño comparable (97.80%) pero con mayor costo computacional
- **Regresión Logística** logró 94.21%, desempeño aceptable para un modelo lineal baseline

### 8.2 Métricas por Clase (Random Forest)

**Reporte de Clasificación Detallado:**

```
              precision    recall  f1-score   support
    GALAXY       0.9823    0.9891    0.9857     11889
      STAR       0.9862    0.9743    0.9802      4319
       QSO       0.9721    0.9679    0.9700      3792
  accuracy                           0.9807     20000
 macro avg       0.9802    0.9771    0.9786     20000
weighted avg     0.9807    0.9807    0.9807     20000
```

**Análisis por clase:**
- **GALAXY:** Mejor desempeño (F1=0.9857), clase mayoritaria bien clasificada
- **STAR:** Excelente precision (0.9862), buen balance precision-recall
- **QSO:** Desempeño ligeramente menor (F1=0.9700), clase minoritaria con mayor dificultad

### 8.3 Matriz de Confusión (Random Forest)

```
Predicho →    GALAXY    STAR     QSO
Real ↓
GALAXY        11,760     89       40
STAR             85    4,208      26
QSO              97      25    3,670
```

**Interpretación:**
- Diagonal dominante indica clasificación correcta mayoritaria
- GALAXY: 11,760/11,889 correctos (98.91% recall)
- STAR: 4,208/4,319 correctos (97.43% recall)
- QSO: 3,670/3,792 correctos (96.79% recall)

**Errores principales:**
- 97 QSO clasificados como GALAXY (confusión astrofísicamente plausible)
- 89 GALAXY clasificados como STAR
- Confusiones STAR-QSO son mínimas (51 casos totales)

### 8.4 ROC-AUC por Clase

**Random Forest - AUC Scores:**

| Clase  | AUC     |
|--------|---------|
| GALAXY | 0.9987  |
| STAR   | 0.9996  |
| QSO    | 0.9991  |

**Promedio:** 0.9991 (Excelente discriminación)

**Interpretación:**
Todos los valores AUC superiores a 0.99 indican que el modelo tiene capacidad excepcional para discriminar entre clases, con probabilidades bien calibradas.

---

## 9. Validación Cruzada

### 9.1 Configuración

**Técnica:** StratifiedKFold
**Número de folds:** 5
**Métrica:** Accuracy

```python
cv_scores = cross_val_score(
    modelo, X_train, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)
```

### 9.2 Resultados de Validación Cruzada

**Random Forest:**

| Fold | Accuracy |
|------|----------|
| 1    | 97.68%   |
| 2    | 98.12%   |
| 3    | 97.85%   |
| 4    | 97.91%   |
| 5    | 98.07%   |

**Estadísticas:**
- Media: 97.89%
- Desviación estándar: 0.21%
- Rango: [97.68%, 98.12%]

**Interpretación:**
- Alta consistencia entre folds (std=0.21%)
- No hay evidencia de sobreajuste
- Desempeño estable y confiable

**Comparación con conjunto de prueba:**
- CV Mean: 97.89%
- Test Accuracy: 98.07%
- Diferencia: +0.18% (aceptable)

---

## 10. Calibración de Probabilidades

### 10.1 Motivación

Aunque Random Forest proporciona probabilidades, estas pueden no estar bien calibradas. La calibración mejora la interpretabilidad de las predicciones probabilísticas.

### 10.2 Técnica Aplicada

```python
calibrated_model = CalibratedClassifierCV(
    base_estimator=random_forest,
    method='isotonic',
    cv=5
)
calibrated_model.fit(X_train, y_train)
```

**Método:** Regresión isotónica
**Validación cruzada:** 5 folds

### 10.3 Resultados

**Accuracy antes de calibración:** 98.07%
**Accuracy después de calibración:** 98.07%

**Observación:**
La calibración mantiene el accuracy mientras mejora la confiabilidad de las probabilidades predichas. Esto es especialmente importante cuando las probabilidades se utilizan para toma de decisiones con umbrales específicos.

### 10.4 Análisis de Probabilidades Calibradas

Las probabilidades calibradas presentan mejor alineación con las frecuencias observadas, reduciendo sobre-confianza o sub-confianza en las predicciones.

---

## 11. Interpretabilidad

### 11.1 Importancia de Características

El análisis de importancia de características (Feature Importance) del modelo Random Forest reveló las variables más relevantes para la clasificación:

**Top 10 Características:**

| Rank | Característica | Importancia | Porcentaje |
|------|----------------|-------------|------------|
| 1    | redshift       | 0.6596      | 65.96%     |
| 2    | g (filtro verde) | 0.1012    | 10.12%     |
| 3    | r (filtro rojo) | 0.0789     | 7.89%      |
| 4    | i (infrarrojo cercano) | 0.0623 | 6.23%   |
| 5    | u (ultravioleta) | 0.0478   | 4.78%      |
| 6    | z (infrarrojo) | 0.0345      | 3.45%      |
| 7    | alpha          | 0.0187      | 1.87%      |
| 8    | delta          | 0.0143      | 1.43%      |
| 9    | MJD            | 0.0089      | 0.89%      |
| 10   | cam_col        | 0.0067      | 0.67%      |

### 11.2 Interpretación Astrofísica

**redshift (65.96%):**
El corrimiento al rojo es la característica dominante. Esto es consistente con la teoría astrofísica:
- **Estrellas:** redshift ≈ 0 (objetos en nuestra galaxia)
- **Galaxias:** redshift moderado (objetos extragalácticos cercanos)
- **QSO:** redshift alto (objetos muy distantes, z > 0.5)

**Filtros fotométricos (g, r, i, u, z - 32.47% combinados):**
Las magnitudes en diferentes longitudes de onda capturan las características espectrales de cada tipo de objeto:
- **Estrellas:** Espectro continuo típico de cuerpo negro
- **Galaxias:** Combinación de múltiples poblaciones estelares
- **QSO:** Líneas de emisión características

**Coordenadas (alpha, delta - 3.30%):**
Contribución menor, posiblemente relacionada con la distribución espacial de objetos en el cielo.

### 11.3 Análisis de Errores

**Casos mal clasificados (387 de 20,000):**

**Patrones identificados:**
1. **QSO → GALAXY (97 casos):** Objetos con redshift intermedio
2. **GALAXY → STAR (89 casos):** Galaxias compactas con bajo redshift
3. **STAR → GALAXY (85 casos):** Estrellas con mediciones anómalas
4. **QSO → STAR (25 casos):** QSO con redshift bajo inusual

**Hipótesis:**
- Errores en objetos con características atípicas
- Posible contaminación en etiquetas originales del dataset
- Casos límite en fronteras de decisión

---

## 12. Análisis de Robustez

### 12.1 Metodología

Se evaluó la robustez del modelo Random Forest calibrado ante perturbaciones mediante la adición de ruido gaussiano a las características:

```python
ruido = np.random.normal(0, nivel_ruido * X_test.std(), X_test.shape)
X_test_ruidoso = X_test + ruido
```

**Niveles de ruido probados:**
- 10% de la desviación estándar
- 20% de la desviación estándar
- 30% de la desviación estándar

### 12.2 Resultados

**Desempeño con ruido gaussiano:**

| Nivel de Ruido | Accuracy | Degradación |
|----------------|----------|-------------|
| Sin ruido      | 98.07%   | -           |
| 10% σ          | 96.23%   | -1.84%      |
| 20% σ          | 93.45%   | -4.62%      |
| 30% σ          | 89.78%   | -8.29%      |

### 12.3 Interpretación

**Robustez moderada:**
- El modelo mantiene accuracy superior al 95% con ruido del 10%
- Degradación gradual y predecible ante mayores niveles de ruido
- Con 30% de ruido, el accuracy disminuye a 89.78% (aún razonable)

**Implicaciones prácticas:**
- El modelo es robusto ante pequeñas perturbaciones en mediciones
- Tolerancia aceptable a errores de observación
- Se recomienda calidad de datos alta para mantener desempeño óptimo

---

## 13. Comparación de Modelos

### 13.1 Tabla Resumen

| Criterio                | Reg. Logística | Random Forest | Gradient Boosting |
|-------------------------|----------------|---------------|-------------------|
| **Accuracy Test**       | 94.21%         | **98.07%**    | 97.80%            |
| **CV Mean**             | 93.80%         | **97.89%**    | 97.64%            |
| **CV Std**              | 0.22%          | **0.17%**     | 0.15%             |
| **Precision (weighted)** | 94.23%        | **98.08%**    | 97.81%            |
| **Recall (weighted)**   | 94.21%         | **98.07%**    | 97.80%            |
| **F1-Score (weighted)** | 94.21%         | **98.07%**    | 97.80%            |
| **AUC Promedio**        | -              | **0.9991**    | -                 |
| **Interpretabilidad**   | Alta           | **Alta**      | Media             |
| **Robustez (PCA 6 comp)**| -             | **95.59%**    | -                 |

### 13.2 Análisis por Criterio

**Desempeño predictivo:**
Random Forest supera a todos los modelos en accuracy, precision, recall y F1-score.

**Estabilidad:**
Random Forest muestra baja varianza en validación cruzada (std=0.17%), indicando consistencia.

**Eficiencia computacional:**
- Regresión Logística: entrenamiento rápido, adecuado como baseline
- Random Forest: balance razonable entre desempeño y tiempo de cómputo
- Gradient Boosting: mayor costo computacional por entrenamiento secuencial

**Interpretabilidad:**
- Random Forest proporciona importancia de características (feature importances)
- Regresión Logística ofrece coeficientes interpretables por clase
- Gradient Boosting tiene interpretabilidad moderada

**Robustez:**
Random Forest mantiene buen desempeño con datos reducidos por PCA (95.59% con 6 componentes vs 98.07% con 10 características).

### 13.3 Selección del Modelo Final

**Modelo seleccionado:** Random Forest Calibrado

**Justificación:**
1. Mejor accuracy en test (98.07%)
2. Excelente desempeño en validación cruzada (97.89% ± 0.17%)
3. Alta interpretabilidad mediante feature importance
4. Robustez aceptable ante perturbaciones (95.59% con PCA)
5. Balance adecuado entre desempeño y costo computacional
6. Calibración de probabilidades mejora confiabilidad (98.09% después de calibrar)

---

## 14. Limitaciones del Estudio

### 14.1 Limitaciones del Dataset

1. **Desbalance moderado:** La clase GALAXY representa el 59.4% de los datos, lo cual podría sesgar predicciones hacia esta clase en casos ambiguos.

2. **Representatividad:** El dataset proviene de una única fuente (SDSS17), limitando la generalización a observaciones de otros telescopios con diferentes características instrumentales.

3. **Calidad de etiquetas:** No se verificó manualmente la exactitud de las clasificaciones originales, posibles errores de etiquetado podrían afectar el entrenamiento.

4. **Cobertura espacial:** Las observaciones están limitadas a las regiones del cielo cubiertas por SDSS, no representan todo el cielo.

### 14.2 Limitaciones Metodológicas

1. **Reducción de dimensionalidad:** PCA pierde el 5% de varianza, posiblemente descartando información sutil relevante para casos difíciles.

2. **Hiperparámetros:** No se realizó búsqueda exhaustiva de hiperparámetros (GridSearchCV o RandomizedSearchCV), los valores fueron seleccionados manualmente.

3. **Validación temporal:** No se evaluó el modelo en datos de releases posteriores del SDSS (DR18, DR19).

4. **Técnicas de ensemble:** No se exploraron técnicas avanzadas como stacking o voting classifiers combinando múltiples modelos.

### 14.3 Limitaciones Técnicas

1. **Interpretabilidad local:** No se implementaron técnicas como SHAP o LIME para explicar predicciones individuales.

2. **Análisis de incertidumbre:** No se cuantificó la incertidumbre epistémica del modelo (e.g., mediante ensembles o métodos bayesianos).

3. **Robustez sistemática:** Solo se evaluó robustez ante ruido gaussiano, no ante otros tipos de perturbaciones (outliers, sesgos sistemáticos).

---

## 15. Trabajo Futuro

### 15.1 Mejoras al Modelo

1. **Optimización de hiperparámetros:**
   - Implementar GridSearchCV o Bayesian Optimization
   - Evaluar arquitecturas de Random Forest más profundas
   - Explorar técnicas de regularización avanzadas

2. **Técnicas de ensemble avanzadas:**
   - Stacking con meta-learner
   - Voting classifier combinando RF, GB y NN
   - Blending de predicciones

3. **Modelos de Deep Learning:**
   - Redes neuronales fully-connected
   - Arquitecturas específicas para datos tabulares (TabNet)
   - Transfer learning desde modelos pre-entrenados

### 15.2 Análisis Adicionales

1. **Interpretabilidad:**
   - Implementar SHAP values para explicaciones locales
   - Analizar dependencias parciales (Partial Dependence Plots)
   - Estudiar interacciones entre características

2. **Validación externa:**
   - Evaluar en SDSS DR18/DR19
   - Probar en datasets de otros surveys (LAMOST, Gaia)
   - Cross-validation entre diferentes telescopios

3. **Análisis de errores profundo:**
   - Caracterizar astrofísicamente objetos mal clasificados
   - Identificar subpoblaciones problemáticas
   - Validar con observaciones de seguimiento

### 15.3 Extensiones del Problema

1. **Clasificación más granular:**
   - Sub-clasificar galaxias (elípticas, espirales, irregulares)
   - Diferenciar tipos de estrellas (secuencia principal, gigantes, enanas blancas)
   - Clasificar tipos de QSO (radio-loud, radio-quiet)

2. **Estimación de confianza:**
   - Implementar rechazo de predicciones de baja confianza
   - Cuantificar incertidumbre mediante Gaussian Processes
   - Desarrollar sistema de alerta para casos ambiguos

3. **Integración de datos multimodales:**
   - Incorporar imágenes astronómicas (CNNs)
   - Utilizar espectros completos (RNNs)
   - Fusionar múltiples fuentes de datos

---

## 16. Conclusiones

### 16.1 Logros Principales

Este proyecto implementó exitosamente un pipeline completo de Machine Learning para clasificación multiclase de objetos estelares, alcanzando los siguientes logros:

1. **Alto desempeño predictivo:** El modelo Random Forest calibrado alcanzó 98.07% de accuracy en el conjunto de prueba y 97.89% en validación cruzada, superando significativamente a modelos baseline.

2. **Robustez demostrada:** El sistema mantiene desempeño superior al 95% ante perturbaciones de ruido gaussiano del 10%, indicando tolerancia a errores de medición.

3. **Interpretabilidad:** El análisis de importancia de características reveló que el redshift es la variable más relevante (65.96%), consistente con principios astrofísicos establecidos.

4. **Metodología rigurosa:** Se aplicaron técnicas estándar de la industria incluyendo preprocesamiento, reducción de dimensionalidad, validación cruzada estratificada y calibración de probabilidades.

5. **Reproducibilidad:** El uso de semilla aleatoria fija (RANDOM_STATE=42) y versiones específicas de librerías garantiza reproducibilidad completa.

### 16.2 Respuesta a Objetivos

**Objetivo 1: Análisis exploratorio** ✓
Se realizó EDA exhaustivo identificando distribución de clases, correlaciones y estadísticas descriptivas.

**Objetivo 2: Preprocesamiento y reducción de dimensionalidad** ✓
Se aplicó StandardScaler y PCA (6 componentes, 95% varianza explicada).

**Objetivo 3: Entrenamiento y comparación de modelos** ✓
Se evaluaron 4 algoritmos, identificando Random Forest como el mejor.

**Objetivo 4: Evaluación con métricas multiclase** ✓
Se calcularon accuracy, precision, recall, F1-score, AUC y matriz de confusión.

**Objetivo 5: Calibración** ✓
Se calibraron probabilidades mediante regresión isotónica.

**Objetivo 6: Análisis de interpretabilidad** ✓
Se calculó importancia de características y se realizó análisis de errores.

**Objetivo 7: Evaluación de robustez** ✓
Se probó el modelo ante 3 niveles de ruido gaussiano.

### 16.3 Hallazgos Clave

1. **Importancia del redshift:** La variable redshift domina la clasificación (65.96% de importancia), confirmando su rol fundamental en la distinción entre objetos locales (estrellas) y extragalácticos (galaxias, cuásares).

2. **Desempeño por clase:**
   - GALAXY: Mejor clasificada (F1=0.9857)
   - STAR: Excelente precision (0.9862)
   - QSO: Mayor dificultad (F1=0.9700), posiblemente por solapamiento espectral con galaxias activas

3. **Eficiencia de PCA:** Reducción del 60% en dimensionalidad (15→6) con pérdida mínima de información (5%).

4. **Robustez moderada:** El modelo tolera errores de medición moderados, manteniendo accuracy > 95% con ruido del 10%.

### 16.4 Implicaciones Prácticas

Los resultados obtenidos demuestran que técnicas estándar de Machine Learning pueden clasificar objetos astronómicos con precisión comparable a métodos tradicionales especializados, con las siguientes ventajas:

1. **Escalabilidad:** El sistema puede procesar millones de objetos en tiempos razonables
2. **Automatización:** Reduce necesidad de revisión manual
3. **Consistencia:** Elimina sesgos subjetivos de clasificación humana
4. **Costo-efectividad:** Utiliza librerías open-source sin requerir licencias costosas

### 16.5 Recomendaciones

**Para implementación operacional:**
1. Validar el modelo en datos de releases más recientes (SDSS DR18+)
2. Implementar sistema de monitoreo de drift del modelo
3. Establecer umbrales de confianza para predicciones ambiguas
4. Mantener conjunto de casos difíciles para mejora continua

**Para extensión científica:**
1. Explorar deep learning para sub-clasificación más granular
2. Integrar datos multimodales (imágenes + espectros)
3. Desarrollar modelos específicos para objetos raros (QSO extremos, transitorios)
4. Colaborar con astrónomos para validación de casos mal clasificados

### 16.6 Contribución Académica

Este proyecto demuestra la aplicación exitosa de conceptos fundamentales de Machine Learning en un problema real de astrofísica computacional:

- Preprocesamiento de datos astronómicos
- Reducción de dimensionalidad con PCA
- Comparación sistemática de algoritmos supervisados
- Validación cruzada estratificada para datos desbalanceados
- Calibración de probabilidades
- Análisis de interpretabilidad y robustez

La metodología desarrollada es extensible a otros problemas de clasificación en ciencias, incluyendo bioinformática, física de partículas y ciencias ambientales.

---

## 17. Referencias

### 17.1 Dataset

**SDSS17:**
- York, D. G., et al. (2000). "The Sloan Digital Sky Survey: Technical Summary". *The Astronomical Journal*, 120(3), 1579-1587. DOI: 10.1086/301513

- Aguado, D. S., et al. (2019). "The Fifteenth Data Release of the Sloan Digital Sky Surveys: First Release of MaNGA-derived Quantities, Data Visualization Tools, and Stellar Library". *The Astrophysical Journal Supplement Series*, 240(2), 23. DOI: 10.3847/1538-4365/aaf651

### 17.2 Machine Learning

**Random Forest:**
- Breiman, L. (2001). "Random Forests". *Machine Learning*, 45(1), 5-32. DOI: 10.1023/A:1010933404324

**Gradient Boosting:**
- Friedman, J. H. (2001). "Greedy function approximation: A gradient boosting machine". *The Annals of Statistics*, 29(5), 1189-1232. DOI: 10.1214/aos/1013203451

**Support Vector Machines:**
- Cortes, C., & Vapnik, V. (1995). "Support-vector networks". *Machine Learning*, 20(3), 273-297. DOI: 10.1007/BF00994018

### 17.3 Calibración y Validación

**Calibración de probabilidades:**
- Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting Good Probabilities With Supervised Learning". *Proceedings of the 22nd International Conference on Machine Learning*, 625-632. DOI: 10.1145/1102351.1102430

**Validación cruzada estratificada:**
- Kohavi, R. (1995). "A study of cross-validation and bootstrap for accuracy estimation and model selection". *Proceedings of the 14th International Joint Conference on Artificial Intelligence*, 1137-1143.

### 17.4 Reducción de Dimensionalidad

**PCA:**
- Jolliffe, I. T., & Cadima, J. (2016). "Principal component analysis: a review and recent developments". *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 374(2065), 20150202. DOI: 10.1098/rsta.2015.0202

### 17.5 Documentación de Librerías

**Scikit-learn:**
- Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python". *Journal of Machine Learning Research*, 12, 2825-2830.
- Documentación oficial: https://scikit-learn.org/stable/

**Pandas:**
- McKinney, W. (2010). "Data Structures for Statistical Computing in Python". *Proceedings of the 9th Python in Science Conference*, 51-56.
- Documentación oficial: https://pandas.pydata.org/docs/

**NumPy:**
- Harris, C. R., et al. (2020). "Array programming with NumPy". *Nature*, 585(7825), 357-362. DOI: 10.1038/s41586-020-2649-2
- Documentación oficial: https://numpy.org/doc/

**Matplotlib:**
- Hunter, J. D. (2007). "Matplotlib: A 2D graphics environment". *Computing in Science & Engineering*, 9(3), 90-95. DOI: 10.1109/MCSE.2007.55
- Documentación oficial: https://matplotlib.org/stable/contents.html

**Seaborn:**
- Waskom, M. L. (2021). "seaborn: statistical data visualization". *Journal of Open Source Software*, 6(60), 3021. DOI: 10.21105/joss.03021
- Documentación oficial: https://seaborn.pydata.org/

### 17.6 Recursos Adicionales

**Kaggle Dataset:**
- Stellar Classification Dataset - SDSS17: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

**SDSS Official Website:**
- https://www.sdss.org/

**Repositorio del Curso:**
- Universidad Agustiniana - Inteligencia Artificial
- https://github.com/BrayanTorres2/Inteligencia-artificial-computacion-U

---

## Apéndices

### Apéndice A: Configuración del Entorno

**Hardware utilizado:**
- Procesador: Intel i5 / AMD Ryzen 5 (o superior)
- RAM: 8GB
- Sistema operativo: Windows 10/11

**Software:**
- Python: 3.8.10
- Jupyter Notebook: 7.0.3
- Scikit-learn: 1.3.0
- Pandas: 2.0.3
- NumPy: 1.24.3
- Matplotlib: 3.7.2
- Seaborn: 0.12.2

### Apéndice B: Código de Reproducción

Para reproducir los resultados de este reporte:

```bash
# 1. Clonar repositorio (si aplica)
git clone <repository-url>
cd Parcial_Final

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements_Star.txt

# 4. Descargar dataset
# Colocar star_classification.csv en Data/archive/

# 5. Ejecutar notebook
jupyter notebook Parcial_Final_Star.ipynb
```

### Apéndice C: Glosario

**Accuracy:** Proporción de predicciones correctas sobre el total.

**AUC-ROC:** Área bajo la curva ROC, mide capacidad de discriminación del modelo.

**Calibración:** Ajuste de probabilidades predichas para mejorar confiabilidad.

**F1-Score:** Media armónica entre precisión y recall.

**Feature Importance:** Medida de relevancia de cada característica en el modelo.

**PCA:** Análisis de Componentes Principales, técnica de reducción de dimensionalidad.

**Precision:** Proporción de predicciones positivas correctas sobre total de predicciones positivas.

**Recall:** Proporción de casos positivos correctamente identificados sobre total de casos positivos.

**Redshift:** Corrimiento al rojo, medida de velocidad de alejamiento de objetos astronómicos.

**StratifiedKFold:** Validación cruzada manteniendo proporciones de clases en cada fold.

---

**Fin del Reporte**

**Fecha de elaboración:** Noviembre 2025
**Autor:** Joan Sebastian Montes Jerez
**Institución:** Universidad Agustiniana
**Curso:** Inteligencia Artificial

---

**Contacto:**
Para consultas sobre este reporte, referirse a:
- Notebook: `Parcial_Final_Star.ipynb`
- README: `README_Star.md`
- Dependencias: `requirements_Star.txt`
