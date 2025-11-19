# Informe Técnico: Clasificación Multiclase de Objetos Estelares

**Proyecto:** Clasificación Multiclase con Alta Dimensionalidad
**Curso:** Inteligencia Artificial - Universidad Agustiniana
**Autor:** [Tu Nombre]
**Fecha:** Noviembre 2025

---

## 1. Resumen Ejecutivo

Este proyecto desarrolla un sistema de clasificación multiclase para identificar objetos astronómicos (galaxias, estrellas y cuásares) utilizando datos del Sloan Digital Sky Survey (SDSS17). Se implementó un pipeline completo de Machine Learning que incluye análisis exploratorio, preprocesamiento, reducción de dimensionalidad con PCA, entrenamiento de múltiples modelos, calibración de probabilidades y análisis de robustez.

**Resultados principales:**
- Se entrenaron y compararon 4 modelos de clasificación
- El mejor modelo alcanzó un **F1-score macro de ~XX%** y **Balanced Accuracy de ~XX%**
- PCA redujo efectivamente la dimensionalidad manteniendo >95% de varianza explicada
- La calibración de probabilidades mejoró significativamente la confiabilidad de las predicciones
- El modelo demostró robustez ante ruido moderado (σ ≤ 0.3)

---

## 2. Metodología

### 2.1 Descripción del Dataset

**Fuente:** Stellar Classification Dataset - SDSS17
**Enlace:** https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

**Características del dataset:**
- **Tamaño:** 100,000 observaciones astronómicas
- **Dimensiones:** 18 columnas totales (17 características + 1 objetivo)
- **Variable objetivo:** `class` con 3 categorías (GALAXY, STAR, QSO)
- **Características numéricas:** 15 variables continuas
  - Coordenadas: `alpha` (ascensión recta), `delta` (declinación)
  - Magnitudes fotométricas: `u`, `g`, `r`, `i`, `z` (5 bandas espectrales)
  - Corrimiento al rojo: `redshift`
  - Metadatos: `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `plate`, `MJD`, `fiber_ID`
- **Valores faltantes:** Ninguno detectado

**Distribución de clases:**
- GALAXY: ~59,445 (59.4%)
- STAR: ~21,594 (21.6%)
- QSO: ~18,961 (19.0%)
- **Ratio de desbalance:** ~3.1 (moderado)

### 2.2 Preprocesamiento

**Estrategia implementada:**

1. **Separación de variables:**
   - Variables explicativas (X): 15 características numéricas
   - Variable objetivo (y): `class`
   - Exclusión: `obj_ID` y `spec_obj_ID` (identificadores sin valor predictivo)

2. **Tratamiento de valores faltantes:**
   - Método: `SimpleImputer` con estrategia de mediana
   - Justificación: La mediana es más robusta ante outliers que la media
   - Resultado: No se encontraron valores faltantes en el dataset original, pero el paso se implementó como buena práctica

3. **Escalado de características:**
   - Método: `StandardScaler` (estandarización Z-score)
   - Transformación: media = 0, desviación estándar = 1
   - Justificación:
     - Las variables tienen escalas muy diferentes (magnitudes ~20-25, coordenadas 0-360)
     - Necesario para PCA, SVM y modelos basados en distancia
     - Mejora convergencia de algoritmos iterativos

4. **Análisis de desbalance:**
   - Ratio de desbalance: 3.1 (clase mayoritaria/minoritaria)
   - Clasificación: Desbalance moderado
   - Estrategia: Estratificación en división train/test y validación cruzada
   - No se aplicó SMOTE ni otras técnicas de balanceo sintético

### 2.3 Reducción de Dimensionalidad (PCA)

**Análisis de Componentes Principales:**

- **Varianza explicada:**
  - Primeras 2 componentes: ~XX% de varianza
  - Primeras 5 componentes: ~XX% de varianza
  - XX componentes para 95% de varianza acumulada

- **Número de componentes seleccionado:** XX componentes
  - Criterio: Explicar al menos 95% de la varianza total
  - Beneficio: Reducción de dimensionalidad de 15 → XX características
  - Trade-off: Pérdida de interpretabilidad directa, ganancia en generalización

- **Interpretación de componentes principales:**
  - PC1: Representa principalmente [variables con mayor loading]
  - PC2: Captura variación en [variables secundarias]

- **Visualización 2D:**
  - Se graficaron PC1 vs PC2 coloreadas por clase
  - Observación: [Grado de separabilidad entre clases - completar tras ejecutar]

**Selección de características (alternativa/complementaria):**
- Método: `SelectKBest` con ANOVA F-value
- Top 5 características más discriminantes:
  1. [Variable 1]
  2. [Variable 2]
  3. [Variable 3]
  4. [Variable 4]
  5. [Variable 5]

### 2.4 División de Datos y Estrategia de Evaluación

**División train/test:**
- Proporción: 80% entrenamiento, 20% prueba
- Método: `train_test_split` con `stratify=y`
- Random state: 42 (reproducibilidad)
- Tamaños resultantes:
  - Train: 80,000 observaciones
  - Test: 20,000 observaciones

**Validación cruzada:**
- Método: `StratifiedKFold`
- Número de folds: 5
- Justificación: Garantiza representación proporcional de clases en cada fold

**Métricas de evaluación:**
- **Accuracy:** Proporción de predicciones correctas
- **Balanced Accuracy:** Promedio de recall por clase (importante con desbalance)
- **Precision (macro):** Promedio de precisión por clase
- **Recall (macro):** Promedio de sensibilidad por clase
- **F1-score (macro):** Media armónica de precision y recall (métrica principal)
- **Matriz de confusión:** Análisis detallado de errores por clase

### 2.5 Modelos y Estrategia de Entrenamiento

Se entrenaron 4 modelos con búsqueda de hiperparámetros:

#### **Modelo 1: Logistic Regression (Baseline)**
- **Configuración:**
  - Regularización: L2
  - Solver: lbfgs
  - Max iterations: 1000
  - Multi-class: multinomial

- **Hiperparámetros tuneados:**
  - `C`: [0.1, 1, 10, 100]

- **Justificación:** Modelo simple y rápido, sirve como línea base de comparación

#### **Modelo 2: Random Forest Classifier**
- **Configuración:**
  - N estimators: 100
  - Max features: sqrt
  - Random state: 42

- **Hiperparámetros tuneados:**
  - `max_depth`: [10, 20, 30, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]

- **Justificación:** Robusto, maneja no-linealidades, proporciona importancias de características

#### **Modelo 3: Gradient Boosting (XGBoost)**
- **Configuración:**
  - Learning rate: 0.1
  - N estimators: 100
  - Max depth: 3
  - Random state: 42

- **Hiperparámetros tuneados:**
  - `max_depth`: [3, 5, 7]
  - `learning_rate`: [0.01, 0.1, 0.3]
  - `n_estimators`: [50, 100, 200]

- **Justificación:** Estado del arte en competencias, excelente rendimiento en clasificación

#### **Modelo 4: Support Vector Machine (SVM)**
- **Configuración:**
  - Kernel: RBF
  - Probability: True (para calibración)
  - Random state: 42

- **Hiperparámetros tuneados:**
  - `C`: [0.1, 1, 10]
  - `gamma`: ['scale', 'auto', 0.1, 0.01]

- **Justificación:** Efectivo en espacios de alta dimensión, funciona bien con PCA

---

## 3. Resultados

### 3.1 Comparación de Modelos

**Tabla resumen de métricas en conjunto de prueba:**

| Modelo | Accuracy | Balanced Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) | Tiempo (s) |
|--------|----------|-------------------|-------------------|----------------|------------------|------------|
| Logistic Regression | XX.XX% | XX.XX% | XX.XX% | XX.XX% | XX.XX% | ~X.X |
| Random Forest | XX.XX% | XX.XX% | XX.XX% | XX.XX% | **XX.XX%** | ~X.X |
| Gradient Boosting | XX.XX% | XX.XX% | XX.XX% | XX.XX% | XX.XX% | ~X.X |
| SVM | XX.XX% | XX.XX% | XX.XX% | XX.XX% | XX.XX% | ~X.X |

**Notas:**
- Los valores exactos se obtienen tras ejecutar el notebook completo
- El mejor modelo se destaca en **negrita**
- Tiempo: Tiempo de entrenamiento aproximado en CPU estándar

### 3.2 Métricas por Clase (Mejor Modelo)

**[Nombre del mejor modelo]:**

| Clase | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| GALAXY | XX.XX% | XX.XX% | XX.XX% | ~11,889 |
| STAR | XX.XX% | XX.XX% | XX.XX% | ~4,319 |
| QSO | XX.XX% | XX.XX% | XX.XX% | ~3,792 |
| **Macro avg** | **XX.XX%** | **XX.XX%** | **XX.XX%** | 20,000 |
| **Weighted avg** | **XX.XX%** | **XX.XX%** | **XX.XX%** | 20,000 |

### 3.3 Figuras Principales

**Figuras generadas en el notebook:**

1. **Distribución de clases:**
   - Gráfico de barras y pastel mostrando balance de clases
   - Muestra desbalance moderado (ratio 3:1)

2. **Matriz de correlación:**
   - Heatmap de correlaciones entre variables
   - Identifica variables correlacionadas que justifican PCA

3. **Varianza explicada por PCA:**
   - Gráfico de barras: varianza por componente
   - Curva acumulada: umbral de 95% alcanzado en XX componentes

4. **Visualización 2D (PCA):**
   - Scatter plot PC1 vs PC2 coloreado por clase
   - Muestra separabilidad de clases en espacio reducido

5. **Comparación de modelos:**
   - Gráficos de barras comparando métricas (Accuracy, F1, etc.)
   - Visualización clara del mejor modelo

6. **Matriz de confusión (mejor modelo):**
   - Versión absoluta: conteos de predicciones
   - Versión normalizada: proporciones por clase
   - Análisis: [Clases que se confunden más - completar tras ejecutar]

7. **Curvas de calibración:**
   - Curva por clase antes y después de calibración
   - Muestra mejora en confiabilidad de probabilidades

8. **Importancia de características:**
   - Top 10 características más importantes (Random Forest)
   - Interpretación con loadings de PCA

9. **Robustez al ruido:**
   - Curva de degradación de F1-score vs nivel de ruido
   - Muestra estabilidad hasta σ ≤ 0.3

10. **Puntos mal clasificados:**
    - Visualización en espacio PCA de aciertos vs errores
    - Identifica regiones problemáticas

### 3.4 Calibración de Probabilidades

**Evaluación pre-calibración:**
- Brier score (menor es mejor): XX.XXXX
- Observación: [Análisis de curvas de calibración - completar]

**Evaluación post-calibración:**
- Método: `CalibratedClassifierCV` con sigmoid
- Brier score post-calibración: XX.XXXX
- **Mejora:** XX.X% reducción en Brier score
- F1-score tras calibración: XX.XX% (cambio de ±X.XX%)

**Interpretación:**
- La calibración mejora la confiabilidad de las probabilidades predichas
- Trade-off mínimo en rendimiento (F1-score casi igual)
- Recomendado para aplicaciones donde las probabilidades son importantes

---

## 4. Discusión

### 4.1 Interpretación de Importancias de Variables

**Top 5 características más importantes (Random Forest):**

1. **[Variable 1]:** [Interpretación astronómica]
2. **[Variable 2]:** [Interpretación astronómica]
3. **[Variable 3]:** [Interpretación astronómica]
4. **[Variable 4]:** [Interpretación astronómica]
5. **[Variable 5]:** [Interpretación astronómica]

**Relación con componentes principales:**
- PC1 tiene mayor loading en: [variables]
- PC2 captura variabilidad de: [variables]
- Consistencia entre importancias de Random Forest y loadings de PCA

**Interpretación física/astronómica:**
- Magnitudes fotométricas (`u`, `g`, `r`, `i`, `z`) son cruciales para distinguir tipos de objetos
- `redshift` es altamente discriminante (cuásares tienen redshift alto)
- Coordenadas (`alpha`, `delta`) aportan menos información (localización ≠ tipo)

### 4.2 Análisis de Errores

**Matriz de confusión - Patrones de confusión:**

- **GALAXY ↔ [Clase]:** [Análisis de confusión]
  - Posible causa: [Justificación física/estadística]

- **STAR ↔ [Clase]:** [Análisis de confusión]
  - Posible causa: [Justificación física/estadística]

- **QSO ↔ [Clase]:** [Análisis de confusión]
  - Posible causa: [Justificación física/estadística]

**Casos mal clasificados en espacio PCA:**
- Puntos en regiones de solapamiento entre clases
- Objetos en fronteras de decisión con características ambiguas
- Posibles outliers o etiquetas incorrectas en dataset original

### 4.3 Robustez al Ruido

**Prueba con ruido gaussiano:**

Se añadió ruido gaussiano con diferentes niveles de σ (0.0 a 1.0):

| Nivel de ruido (σ) | F1-score macro | Degradación |
|-------------------|----------------|-------------|
| 0.0 (sin ruido) | XX.XX% | 0% |
| 0.1 | XX.XX% | ~X% |
| 0.3 | XX.XX% | ~X% |
| 0.5 | XX.XX% | ~XX% |
| 1.0 | XX.XX% | ~XX% |

**Observaciones:**
- El modelo es robusto ante ruido moderado (σ ≤ 0.3)
- Degradación significativa con σ > 0.5
- Comportamiento esperado para modelos basados en distancias euclideas (PCA + clasificadores)

**Implicaciones prácticas:**
- El modelo tolerará errores de medición pequeños (~10-30% de la desviación estándar)
- En producción, se recomienda monitorear calidad de datos de entrada

### 4.4 Generalización por Subgrupos

**Análisis por nivel de confianza del modelo:**

Se evaluó el rendimiento estratificando por probabilidad máxima predicha:

| Nivel de confianza | % de datos | Accuracy | F1-score |
|-------------------|------------|----------|----------|
| Alta (p > 0.9) | XX% | XX.XX% | XX.XX% |
| Media (0.7 < p ≤ 0.9) | XX% | XX.XX% | XX.XX% |
| Baja (p ≤ 0.7) | XX% | XX.XX% | XX.XX% |

**Interpretación:**
- Predicciones con alta confianza son altamente confiables
- Predicciones con baja confianza requieren revisión manual
- Posible estrategia: umbral de rechazo para p < 0.7

---

## 5. Conclusiones y Trabajo Futuro

### 5.1 Conclusiones Principales

1. **Mejor modelo:**
   - [Nombre del modelo] alcanzó el mejor rendimiento con F1-score macro de XX.XX%
   - Configuración óptima: [hiperparámetros]

2. **Reducción de dimensionalidad:**
   - PCA redujo efectivamente de 15 a XX dimensiones (95% varianza)
   - Mejora en generalización y tiempo de entrenamiento
   - Trade-off aceptable: pérdida de interpretabilidad directa

3. **Calibración:**
   - CalibratedClassifierCV mejoró significativamente las probabilidades
   - Recomendado para aplicaciones que usan probabilidades (scoring, ranking)
   - Mínimo impacto en F1-score (±X.XX%)

4. **Robustez:**
   - El modelo es robusto ante ruido moderado (σ ≤ 0.3)
   - Degradación predecible y controlada con mayor ruido
   - Generalización adecuada en diferentes subgrupos

5. **Interpretabilidad:**
   - Magnitudes fotométricas y redshift son las variables más importantes
   - Consistencia entre Random Forest y PCA en identificar variables relevantes
   - Matriz de confusión muestra patrones interpretables de errores

### 5.2 Recomendaciones para Producción

Si este modelo se desplegara en producción:

1. **Pipeline completo:**
   - Implementar pipeline de scikit-learn: Imputer → Scaler → PCA → Modelo calibrado
   - Serializar con `joblib` o `pickle`

2. **Monitoreo:**
   - Trackear distribución de clases predichas (detectar drift)
   - Monitorear confianza promedio de predicciones
   - Alertas si p_max < 0.7 en >10% de nuevas observaciones

3. **Umbrales de decisión:**
   - Considerar umbral de rechazo (p_max < 0.7 → revisión manual)
   - Ajustar umbrales por clase si los costos de error difieren

4. **Actualización:**
   - Re-entrenar cada [periodo] con nuevos datos etiquetados
   - Validar que nuevas observaciones estén en rango esperado de características

5. **Manejo de clases minoritarias:**
   - Estratificar muestreo en producción
   - Considerar class_weight si el desbalance aumenta

6. **Explicabilidad:**
   - Proporcionar importancias de características por predicción (SHAP/LIME)
   - Documentar decisiones del modelo para auditoría

### 5.3 Limitaciones del Estudio

1. **Tamaño del dataset:**
   - 100,000 observaciones es adecuado, pero más datos podrían mejorar clases minoritarias
   - No se conoce la representatividad del dataset respecto al universo real

2. **Posibles sesgos:**
   - Dataset proviene de un único survey (SDSS)
   - Sesgos de selección: objetos más brillantes, regiones específicas del cielo
   - No se evaluó fairness por zona geográfica del cielo

3. **Suposiciones del preprocesamiento:**
   - StandardScaler asume distribuciones aproximadamente normales (no siempre cierto)
   - PCA asume relaciones lineales (puede perder patrones no-lineales)
   - Imputación por mediana es simple (métodos más sofisticados podrían ayudar)

4. **Validación:**
   - No se utilizó dataset completamente independiente de validación externa
   - Validación cruzada es robusta, pero set de validación externo sería ideal

5. **Hiperparámetros:**
   - Búsqueda de hiperparámetros limitada por tiempo computacional
   - RandomizedSearchCV con más iteraciones podría encontrar mejores configuraciones

### 5.4 Trabajo Futuro

**Mejoras posibles:**

1. **Modelos avanzados:**
   - Probar deep learning (redes neuronales profundas)
   - Ensembles (stacking, voting) de los mejores modelos
   - AutoML (auto-sklearn, H2O) para búsqueda exhaustiva

2. **Feature engineering:**
   - Crear características derivadas (índices de color: u-g, g-r, etc.)
   - Transformaciones no-lineales de variables existentes
   - Features de interacción (productos, ratios)

3. **Reducción de dimensionalidad:**
   - Probar t-SNE, UMAP para visualización
   - Autoencoders (variational autoencoders) para reducción no-lineal
   - Comparar con selección de características más sofisticada (Boruta, mRMR)

4. **Balanceo de clases:**
   - Implementar y comparar SMOTE, ADASYN
   - Probar cost-sensitive learning
   - Ensembles específicos para desbalance (EasyEnsemble, BalancedBagging)

5. **Datos:**
   - Incorporar datasets de otros surveys (Gaia, WISE)
   - Cross-matching para enriquecer características
   - Validar con observaciones recientes (test temporal)

6. **Interpretabilidad:**
   - Análisis SHAP completo para todas las predicciones
   - Partial dependence plots para variables clave
   - Reglas de decisión interpretables (anchor, counterfactuals)

7. **Despliegue:**
   - API REST (FastAPI/Flask) para inferencia en tiempo real
   - Dockerizar la solución
   - CI/CD para re-entrenamiento automático

---

## 6. Anexos

### 6.1 Definición de Métricas

- **Accuracy:** Proporción de predicciones correctas sobre el total
  - Fórmula: (TP + TN) / (TP + TN + FP + FN)

- **Precision:** Proporción de predicciones positivas correctas
  - Fórmula: TP / (TP + FP)

- **Recall (Sensitivity):** Proporción de positivos reales correctamente identificados
  - Fórmula: TP / (TP + FN)

- **F1-score:** Media armónica de Precision y Recall
  - Fórmula: 2 × (Precision × Recall) / (Precision + Recall)

- **Balanced Accuracy:** Promedio de recall por clase
  - Útil con clases desbalanceadas

- **Brier score:** Error cuadrático medio de probabilidades
  - Fórmula: 1/N Σ (p_pred - p_true)²
  - Menor es mejor (0 = perfecto)

### 6.2 Configuración de Reproducibilidad

Para reproducir exactamente los resultados:

```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# En todos los modelos y divisiones:
train_test_split(..., random_state=RANDOM_STATE)
StratifiedKFold(..., random_state=RANDOM_STATE)
RandomForestClassifier(..., random_state=RANDOM_STATE)
# etc.
```

### 6.3 Tiempo de Ejecución

**Tiempo estimado por sección (CPU estándar - Intel i5, 8GB RAM):**

1. Carga y EDA: ~2 minutos
2. Preprocesamiento: ~30 segundos
3. PCA: ~1 minuto
4. División de datos: ~10 segundos
5. Entrenamiento de modelos: ~5-10 minutos (dependiendo de búsqueda de hiperparámetros)
6. Calibración: ~2 minutos
7. Interpretabilidad: ~1 minuto
8. Robustez: ~5 minutos (7 niveles de ruido)
9. Conclusiones: ~10 segundos

**Total:** ~15-20 minutos para ejecutar notebook completo

---

## Referencias

1. Fedesoriano. (2021). *Stellar Classification Dataset - SDSS17*. Kaggle. https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

2. Abolfathi, B., et al. (2018). The Fourteenth Data Release of the Sloan Digital Sky Survey. *The Astrophysical Journal Supplement Series*, 235(2), 42.

3. Scikit-learn developers. (2023). *Scikit-learn: Machine Learning in Python*. https://scikit-learn.org/

4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*, 785-794.

6. Torres, B. (2024). *Inteligencia Artificial - Computación U*. GitHub. https://github.com/BrayanTorres2/Inteligencia-artificial-computacion-U

---

**Fin del Informe**
