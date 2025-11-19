# Informe Técnico: Clasificación Multiclase de Objetos Estelares

**Proyecto:** Clasificación Multiclase con Alta Dimensionalidad

**Curso:** Inteligencia Artificial - Universidad Agustiniana
**Autor:** Joan Sebastian Montes Jerez
**Fecha:** Noviembre 2024

---

## 1. Resumen Ejecutivo

Este proyecto desarrolla un sistema de clasificación multiclase para identificar objetos astronómicos (galaxias, estrellas y cuásares) utilizando datos del Sloan Digital Sky Survey (SDSS17). Se implementó un pipeline completo de Machine Learning que incluye análisis exploratorio, preprocesamiento, reducción de dimensionalidad con PCA, entrenamiento de múltiples modelos, calibración de probabilidades y análisis de robustez.

**Resultados principales:**
- Se entrenaron y compararon 4 modelos de clasificación
- El mejor modelo (SVM con kernel RBF) alcanzó un **F1-score weighted de 95.31%** y **Balanced Accuracy de 94.06%**
- PCA redujo efectivamente la dimensionalidad de 15 a 9 componentes manteniendo 96.06% de varianza explicada
- La calibración de probabilidades mejoró el Brier score de 0.0492 a 0.0455 (mejora del 7.3%)
- El modelo calibrado (Random Forest) demostró sensibilidad al ruido, con degradación significativa para σ > 0.3

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
  - Primeras 2 componentes: 48.08% de varianza
  - Primeras 5 componentes: 73.78% de varianza
  - 9 componentes para 96.06% de varianza acumulada

- **Número de componentes seleccionado:** 9 componentes
  - Criterio: Explicar al menos 95% de la varianza total
  - Beneficio: Reducción de dimensionalidad de 15 → 9 características (40% de reducción)
  - Trade-off: Pérdida de interpretabilidad directa, ganancia en generalización

- **Interpretación de componentes principales:**
  - PC1 (26.92% varianza): Representa principalmente MJD (0.454), magnitudes i (0.454) y r (0.450)
  - PC2 (21.16% varianza): Captura variación en magnitudes u (0.565), z (0.564) y g (0.564)
  - PC9 (4.69% varianza): Altamente influenciado por redshift (0.723), crítico para discriminar clases

- **Visualización 2D:**
  - Se graficaron PC1 vs PC2 coloreadas por clase
  - Observación: Se observa separabilidad moderada entre clases, con cierto solapamiento en las fronteras de decisión, especialmente entre GALAXY y QSO

**Selección de características (alternativa/complementaria):**
- Método: Random Forest feature importance basado en componentes principales
- Top 5 componentes más discriminantes:
  1. PC9: 27.03% (principalmente redshift)
  2. PC2: 18.67% (magnitudes u, z, g)
  3. PC1: 12.88% (MJD y magnitudes i, r)
  4. PC6: 10.04%
  5. PC8: 9.79%

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

| Modelo | Accuracy | Balanced Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) | Overfitting |
|--------|----------|-------------------|----------------------|-------------------|---------------------|-------------|
| Logistic Regression | 90.28% | 87.60% | 90.47% | 90.28% | 90.23% | -0.0015 |
| Random Forest | 90.76% | 87.69% | 90.92% | 90.76% | 90.69% | 0.0651 |
| Gradient Boosting | 91.49% | 88.88% | 91.61% | 91.49% | 91.44% | 0.0250 |
| **SVM (RBF)** | **95.34%** | **94.06%** | **95.38%** | **95.34%** | **95.31%** | **-0.0008** |

**Notas:**
- El mejor modelo es **SVM con kernel RBF**, destacado en negrita
- SVM muestra el mejor rendimiento sin overfitting (valor negativo indica ligera mejora en test)
- Random Forest presenta mayor overfitting (0.0651) pero se seleccionó para calibración
- Tiempo: No registrado explícitamente en el notebook

### 3.2 Métricas por Clase (Modelo Calibrado para Producción)

**Random Forest con CalibratedClassifierCV:**

| Clase | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| GALAXY | 91% | 95% | 93% | 11,889 |
| QSO | 96% | 87% | 91% | 3,792 |
| STAR | 87% | 85% | 86% | 4,319 |
| **Macro avg** | **91%** | **89%** | **90%** | 20,000 |
| **Weighted avg** | **91%** | **91%** | **91%** | 20,000 |

**Accuracy global del modelo calibrado:** 91.09%

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
   - Análisis: GALAXY se clasifica correctamente en 95% de los casos, con confusión menor con QSO (4%). STAR tiene mayor dificultad con 15% de error, principalmente confundido con GALAXY. QSO se confunde en 13% con GALAXY

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

**Evaluación pre-calibración (Random Forest):**
- Brier score GALAXY: 0.0710
- Brier score QSO: 0.0281
- Brier score STAR: 0.0483
- **Brier score promedio: 0.0492** (menor es mejor)
- Accuracy: 90.76%

**Evaluación post-calibración:**
- Método: `CalibratedClassifierCV` con sigmoid
- **Brier score post-calibración: 0.0455**
- **Mejora:** 7.3% reducción en Brier score (0.0037 puntos)
- Accuracy tras calibración: 91.09% (mejora de +0.33%)
- F1-score tras calibración: 91.05%

**Interpretación:**
- La calibración mejora significativamente la confiabilidad de las probabilidades predichas
- Trade-off positivo: mejora tanto en calibración como en accuracy
- Recomendado para aplicaciones donde las probabilidades son importantes (ranking, scoring, umbrales de decisión)

---

## 4. Discusión

### 4.1 Interpretación de Importancias de Variables

**Top 5 componentes principales más importantes (Random Forest):**

1. **PC9 (27.03%):** Dominado por redshift (0.723), run_ID (0.365) y r (-0.369). El redshift es crítico para distinguir cuásares de objetos locales
2. **PC2 (18.67%):** Captura magnitudes u, z, g (loadings ~0.56). Representa información de color en diferentes bandas espectrales
3. **PC1 (12.88%):** Influenciado por MJD (0.454), i (0.454), r (0.450). Relacionado con época de observación y magnitudes rojas
4. **PC6 (10.04%):** Componente secundaria que aporta información complementaria
5. **PC8 (9.79%):** Contribuye con patrones específicos de separabilidad entre clases

**Relación con componentes principales:**
- PC1 tiene mayor loading en: MJD (tiempo de observación), magnitudes i y r (bandas rojas)
- PC2 captura variabilidad de: magnitudes u, z, g (colores azul, infrarrojo cercano y verde)
- PC9 es el más discriminante a pesar de tener solo 4.69% de varianza explicada, demostrando que varianza ≠ poder predictivo

**Interpretación física/astronómica:**
- Magnitudes fotométricas (`u`, `g`, `r`, `i`, `z`) son cruciales para distinguir tipos de objetos
- `redshift` es altamente discriminante (cuásares tienen redshift alto)
- Coordenadas (`alpha`, `delta`) aportan menos información (localización ≠ tipo)

### 4.2 Análisis de Errores

**Matriz de confusión - Patrones de confusión:**

- **GALAXY ↔ QSO:** 4% de galaxias mal clasificadas como QSO
  - Posible causa: Galaxias con núcleos activos (AGN) pueden tener espectros similares a cuásares. Solapamiento en redshift moderado

- **STAR ↔ GALAXY:** 15% de estrellas mal clasificadas como galaxias
  - Posible causa: Estrellas débiles o en regiones de alta extinción pueden confundirse con galaxias compactas. Limitación en resolución espacial

- **QSO ↔ GALAXY:** 13% de cuásares mal clasificados como galaxias
  - Posible causa: Cuásares débiles o con redshift bajo pueden ser indistinguibles de galaxias activas. Ambos tienen componentes estelares

**Casos mal clasificados en espacio PCA:**
- Puntos en regiones de solapamiento entre clases, especialmente en fronteras GALAXY-QSO
- Objetos con características ambiguas: estrellas con magnitudes atípicas, galaxias compactas brillantes
- Posibles outliers o errores de etiquetado en dataset original (menos del 5% de errores sistemáticos)

### 4.3 Robustez al Ruido

**Prueba con ruido gaussiano:**

Se añadió ruido gaussiano con diferentes niveles de σ (0.0 a 1.0) a las características escaladas:

| Nivel de ruido (σ) | Accuracy | F1-score | Degradación Accuracy | Degradación F1 |
|-------------------|----------|----------|---------------------|----------------|
| 0.0 (sin ruido) | 91.09% | 91.05% | 0% | 0% |
| 0.1 | 82.46% | 82.62% | 9.5% | 9.3% |
| 0.2 | 72.33% | 72.64% | 20.6% | 20.2% |
| 0.3 | 65.79% | 66.06% | 27.8% | 27.4% |
| 0.5 | 56.14% | 56.34% | 38.4% | 38.1% |
| 0.7 | 49.84% | 49.62% | 45.3% | 45.5% |
| 1.0 | 44.32% | 43.58% | 51.3% | 52.1% |

**Observaciones:**
- El modelo muestra **sensibilidad moderada-alta al ruido**
- Con σ = 0.1, ya se observa degradación del 9.5% en accuracy
- Degradación significativa con σ > 0.3 (más del 27% de pérdida)
- Degradación total con σ = 1.0: **~51% de pérdida en rendimiento**
- Comportamiento esperado para modelos basados en distancias euclideas (PCA + clasificadores)

**Implicaciones prácticas:**
- El modelo requiere datos de alta calidad con ruido mínimo (σ < 0.1 idealmente)
- No es robusto ante errores de medición significativos
- En producción, se recomienda:
  - Validación estricta de calidad de datos de entrada
  - Filtros de outliers y detección de anomalías
  - Monitoreo continuo de la distribución de características

### 4.4 Generalización por Subgrupos

**Análisis por nivel de confianza del modelo:**

Se evaluó el rendimiento estratificando por probabilidad máxima predicha:

| Rango de Confianza | Número de Muestras | % de datos | Accuracy |
|-------------------|-------------------|------------|----------|
| 0.95 - 1.00 (Muy Alta) | 11,652 | 58.3% | 98.29% |
| 0.85 - 0.95 (Alta) | 4,924 | 24.6% | 90.33% |
| 0.70 - 0.85 (Media) | 1,863 | 9.3% | 75.90% |
| 0.50 - 0.70 (Baja) | 1,437 | 7.2% | 58.39% |
| 0.00 - 0.50 (Muy Baja) | 124 | 0.6% | 50.81% |

**Interpretación:**
- **58.3% de predicciones tienen confianza muy alta (>0.95)** con accuracy del 98.29%
- Existe una **correlación fuerte entre confianza y accuracy**: mayor confianza = mayor precisión
- Predicciones con p < 0.70 (7.8% de datos) tienen accuracy inferior al 60%
- **Estrategia recomendada:**
  - Aceptar automáticamente predicciones con p > 0.85 (82.9% de datos, accuracy >90%)
  - Revisar manualmente predicciones con p < 0.85 (17.1% de datos)
  - Rechazar o investigar predicciones con p < 0.50 (0.6% de datos, accuracy ~random)

---

## 5. Conclusiones y Trabajo Futuro

### 5.1 Conclusiones Principales

1. **Mejor modelo:**
   - **SVM con kernel RBF** alcanzó el mejor rendimiento con F1-score weighted de 95.31% y Balanced Accuracy de 94.06%
   - Sin overfitting (-0.0008), demostrando excelente generalización
   - Sin embargo, se seleccionó **Random Forest calibrado** para producción por mejor interpretabilidad y calibración de probabilidades (F1: 91.05%, Accuracy: 91.09%)

2. **Reducción de dimensionalidad:**
   - PCA redujo efectivamente de 15 a 9 dimensiones (96.06% varianza explicada)
   - Reducción del 40% en dimensionalidad mejora tiempo de entrenamiento y reduce overfitting
   - Trade-off aceptable: pérdida de interpretabilidad directa, ganancia en generalización
   - **Hallazgo importante:** PC9 (solo 4.69% varianza) fue el más discriminante, demostrando que varianza ≠ importancia predictiva

3. **Calibración:**
   - CalibratedClassifierCV mejoró significativamente las probabilidades (Brier score: 0.0492 → 0.0455, mejora del 7.3%)
   - Recomendado para aplicaciones que usan probabilidades (scoring, ranking, umbrales de decisión)
   - **Trade-off positivo:** mejora tanto en calibración (+7.3%) como en accuracy (+0.33%)

4. **Robustez:**
   - El modelo muestra **sensibilidad moderada-alta al ruido**: degradación del 9.5% con σ=0.1
   - Con σ > 0.3, degradación supera el 27%, no apto para datos ruidosos
   - **Limitación importante:** Requiere datos de alta calidad para mantener rendimiento
   - Generalización adecuada por subgrupos de confianza: 98.29% accuracy para p>0.95

5. **Interpretabilidad:**
   - **Redshift (vía PC9)** es la variable más discriminante (27.03% importancia), crítico para identificar cuásares
   - Magnitudes fotométricas u, z, g (vía PC2) aportan 18.67% de importancia
   - Matriz de confusión muestra confusión GALAXY↔QSO (núcleos activos) y STAR↔GALAXY (objetos débiles)

### 5.2 Recomendaciones para Producción

Si este modelo se desplegara en producción:

1. **Pipeline completo:**
   - Implementar pipeline de scikit-learn: Imputer → Scaler → PCA → Modelo calibrado
   - Serializar con `joblib` o `pickle`

2. **Monitoreo:**
   - **Control de calidad de entrada:** Validar que σ de ruido < 0.1 (crítico dada la sensibilidad al ruido)
   - Trackear distribución de clases predichas (detectar drift)
   - Monitorear confianza promedio de predicciones (baseline: 58.3% con p>0.95)
   - Alertas si:
     - p_max < 0.85 en >17% de nuevas observaciones (indica degradación)
     - Distribución de características diverge de entrenamiento (detectar drift)

3. **Umbrales de decisión:**
   - Implementar umbral de rechazo basado en análisis de confianza:
     - **Aceptar automáticamente:** p > 0.85 (82.9% de datos, accuracy >90%)
     - **Revisión manual:** 0.50 < p ≤ 0.85 (16.5% de datos)
     - **Rechazar/Investigar:** p ≤ 0.50 (0.6% de datos, accuracy ~random)
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
