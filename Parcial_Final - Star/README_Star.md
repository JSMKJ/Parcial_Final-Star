# Clasificación Multiclase de Objetos Estelares

**Universidad Agustiniana**
**Curso:** Inteligencia Artificial
**Estudiante:** Joan Sebastian Montes Jerez
**Proyecto:** Parcial Final - Clasificación Multiclase
**Fecha:** Noviembre 2025

---

## Descripción del Proyecto

Este proyecto implementa un sistema de clasificación multiclase para objetos astronómicos utilizando técnicas de Machine Learning. Se emplea el dataset SDSS17 (Sloan Digital Sky Survey Data Release 17) para clasificar objetos celestes en tres categorías: galaxias (GALAXY), estrellas (STAR) y cuásares (QSO).

El análisis incluye exploración de datos, preprocesamiento, reducción de dimensionalidad mediante PCA, entrenamiento de múltiples modelos supervisados, calibración de probabilidades, análisis de interpretabilidad y evaluación de robustez ante perturbaciones.

---

## Objetivo

Desarrollar un modelo predictivo capaz de clasificar objetos estelares con alta precisión, aplicando técnicas de preprocesamiento, reducción de dimensionalidad, validación cruzada y análisis de desempeño mediante métricas estándar de clasificación multiclase.

---

## Dataset

### Información General

- **Fuente:** Sloan Digital Sky Survey Data Release 17 (SDSS17)
- **Observaciones:** 100,000 registros
- **Variables:** 18 características
- **Clases:** 3 categorías (GALAXY, STAR, QSO)
- **Valores faltantes:** No se presentan valores faltantes en el dataset

### Distribución de Clases

| Clase  | Frecuencia | Porcentaje |
|--------|------------|------------|
| GALAXY | 59,445     | 59.445%    |
| STAR   | 21,594     | 21.594%    |
| QSO    | 18,961     | 18.961%    |

**Ratio de desbalance:** 3.14:1

### Características Principales

Las variables predictoras incluyen coordenadas astronómicas, magnitudes fotométricas en diferentes filtros, identificadores de observación y corrimiento al rojo (redshift):

- **Coordenadas:** alpha, delta
- **Filtros fotométricos:** u, g, r, i, z
- **Identificadores:** run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID
- **Física:** redshift (corrimiento al rojo)

---

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Jupyter Notebook
- 8GB de RAM recomendados (mínimo 4GB)
- Aproximadamente 500MB de espacio en disco

### Instalación de Dependencias

Se recomienda crear un entorno virtual antes de instalar las dependencias:

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements_Star.txt
```

### Verificación de Instalación

Para verificar que las librerías se instalaron correctamente:

```bash
pip list | grep -E "numpy|pandas|scikit-learn|matplotlib|seaborn"
```

---

## Estructura del Repositorio

```
Parcial_Final/
├── Data/
│   └── archive/
│       └── star_classification.csv     # Dataset SDSS17
│
├── Parcial_Final_Star.ipynb            # Notebook principal
├── requirements_Star.txt               # Dependencias del proyecto
├── README_Star.md                      # Este archivo
└── REPORT_Star.md                      # Reporte técnico de resultados
```

---

## Uso

### Ejecutar el Notebook

1. Asegurarse de que el dataset esté ubicado en `Data/archive/star_classification.csv`
2. Abrir Jupyter Notebook:

```bash
jupyter notebook Parcial_Final_Star.ipynb
```

3. Ejecutar las celdas secuencialmente desde la Sección 0 hasta la Sección 9

### Estructura del Notebook

El notebook se organiza en 10 secciones principales:

**Sección 0: Configuración del Entorno**
- Importación de librerías
- Configuración de parámetros globales (RANDOM_STATE = 42)
- Configuración de visualizaciones

**Sección 1: Análisis Exploratorio de Datos (EDA)**
- Carga del dataset
- Análisis de estructura y tipos de datos
- Estadísticas descriptivas
- Análisis de correlaciones
- Visualizaciones exploratorias

**Sección 2: Preprocesamiento de Datos**
- Separación de variables predictoras y objetivo
- Tratamiento de valores faltantes (imputación con mediana)
- Escalado de características (StandardScaler)
- Análisis de desbalance de clases

**Sección 3: Reducción de Dimensionalidad**
- Aplicación de PCA (Análisis de Componentes Principales)
- Análisis de varianza explicada
- Selección de componentes (6 componentes para 95% de varianza)
- Visualización de componentes principales

**Sección 4: División de Datos**
- División train-test (80% - 20%)
- Aplicación de estratificación para mantener proporciones de clases
- Verificación de distribución en conjuntos

**Sección 5: Modelado**
- Entrenamiento de 3 modelos:
  - Regresión Logística (baseline)
  - Random Forest
  - Gradient Boosting
- Evaluación con métricas multiclase
- Validación cruzada estratificada (5-fold)

**Sección 6: Calibración de Modelos**
- Calibración de probabilidades con CalibratedClassifierCV
- Método isotónico para mejora de predicciones probabilísticas
- Comparación antes y después de calibración

**Sección 7: Interpretabilidad**
- Análisis de importancia de características (Feature Importance)
- Identificación de variables más relevantes
- Análisis de errores de clasificación

**Sección 8: Análisis de Robustez**
- Evaluación con datos reducidos por PCA
- Pruebas con ruido gaussiano (σ = 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
- Análisis de degradación del desempeño
- Comparación de configuraciones

**Sección 9: Conclusiones**
- Resumen de resultados
- Comparación de modelos
- Recomendaciones técnicas
- Limitaciones y trabajo futuro

---

## Resultados Principales

### Desempeño de Modelos

Se entrenaron y evaluaron tres algoritmos de clasificación supervisada. Los resultados en el conjunto de prueba fueron:

| Modelo                  | Accuracy | Precision | Recall | F1-Score |
|-------------------------|----------|-----------|--------|----------|
| Regresión Logística     | 94.21%   | 94.23%    | 94.21% | 94.21%   |
| **Random Forest**       | **98.07%** | **98.08%** | **98.07%** | **98.07%** |
| Gradient Boosting       | 97.80%   | 97.81%    | 97.80% | 97.80%   |

**Mejor modelo:** Random Forest con 98.07% de accuracy y desempeño consistente en validación cruzada (97.89% ± 0.21%).

### Reducción de Dimensionalidad

La aplicación de PCA permitió reducir las 15 características originales a 6 componentes principales, preservando el 95% de la varianza total. Esta reducción mejora la eficiencia computacional sin comprometer significativamente el desempeño del modelo.

### Importancia de Características

El análisis de Random Forest identificó las variables más relevantes para la clasificación:

1. **redshift** - 65.96% de importancia
2. **g** (filtro verde) - 10.12%
3. **r** (filtro rojo) - 7.89%
4. **i** (filtro infrarrojo cercano) - 6.23%
5. **u** (filtro ultravioleta) - 4.78%

El corrimiento al rojo (redshift) es la característica dominante, lo cual es consistente con la teoría astrofísica, dado que galaxias y cuásares presentan valores de redshift significativamente distintos a las estrellas.

### Calibración de Probabilidades

La calibración isotónica mejoró la confiabilidad de las predicciones probabilísticas del modelo Random Forest, permitiendo interpretaciones más precisas de las probabilidades de clase predichas.

### Robustez

El modelo Random Forest demostró robustez moderada ante perturbaciones:
- Con datos reducidos por PCA (6 componentes): 95.59% accuracy (pérdida de 2.48%)
- Ante ruido gaussiano: degradación gradual y predecible con σ entre 0.1 y 1.0
- Mantiene buen desempeño hasta niveles moderados de perturbación

---

## Metodología

El proyecto sigue un flujo de trabajo estándar en Machine Learning:

1. **Carga y exploración** del dataset astronómico
2. **Preprocesamiento** con imputación y escalado
3. **Reducción de dimensionalidad** mediante PCA
4. **División estratificada** de datos (80/20)
5. **Entrenamiento** de múltiples modelos supervisados
6. **Validación cruzada** con 5 folds estratificados
7. **Calibración** de probabilidades del mejor modelo
8. **Análisis de interpretabilidad** mediante feature importance
9. **Evaluación de robustez** ante perturbaciones
10. **Documentación** de resultados y conclusiones

---

## Tecnologías Utilizadas

### Lenguajes y Entornos

- **Python 3.8+** - Lenguaje de programación principal
- **Jupyter Notebook** - Entorno interactivo de desarrollo

### Librerías de Machine Learning

- **scikit-learn** - Algoritmos de clasificación, preprocesamiento, métricas
- **numpy** - Operaciones numéricas y manejo de arrays
- **pandas** - Manipulación y análisis de datos tabulares

### Librerías de Visualización

- **matplotlib** - Generación de gráficos base
- **seaborn** - Visualizaciones estadísticas avanzadas

### Utilidades

- **joblib** - Serialización de modelos
- **tqdm** - Barras de progreso
- **ipywidgets** - Widgets interactivos en Jupyter

---

## Reproducibilidad

El proyecto garantiza reproducibilidad mediante:

- Semilla aleatoria fija: `RANDOM_STATE = 42`
- Versiones específicas de dependencias en `requirements_Star.txt`
- Dataset público y versionado (SDSS17)
- Código completamente documentado
- Orden de ejecución secuencial explícito

---

## Referencias

### Dataset

- **SDSS17:** Sloan Digital Sky Survey Data Release 17
- **Kaggle:** [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)

### Documentación Técnica

- **Scikit-learn:** [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **Pandas:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- **Matplotlib:** [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
- **Seaborn:** [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

### Referencias Académicas

- **SDSS:** York, D. G., et al. (2000). The Sloan Digital Sky Survey: Technical Summary. *The Astronomical Journal*, 120(3), 1579-1587.
- **Random Forest:** Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- **PCA:** Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065).

---

## Soporte

Para consultas técnicas sobre este proyecto:

- Revisar la documentación en `REPORT_Star.md`
- Consultar comentarios en el notebook `Parcial_Final_Star.ipynb`
- Verificar la instalación de dependencias en `requirements_Star.txt`

---

## Licencia

Este proyecto fue desarrollado con fines académicos para el curso de Inteligencia Artificial de la Universidad Agustiniana.

---

**Última actualización:** Noviembre 2025
**Versión:** 1.0