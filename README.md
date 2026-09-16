# Diabetes Classification Model

Modelo de clasificación binaria para predecir la presencia de diabetes a partir de indicadores clínicos, usando el dataset Pima Indians Diabetes.

## 📊 Dataset

- **Fuente**: [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv)
- **Registros**: 768 pacientes
- **Variables**: 8 features clínicas (Glucose, BMI, Age, Insulin, BloodPressure, SkinThickness, Pregnancies, DiabetesPedigreeFunction) + 1 target (Outcome)
- **Balance de clases**: 500 sin diabetes (65%) / 268 con diabetes (35%)

## 🔧 Pipeline del proyecto

1. **EDA**: distribución de clases, correlación entre variables, identificación de outliers y sesgo (skew)
2. **Limpieza de datos**: detección de ceros fisiológicamente inválidos (Glucose, BloodPressure, SkinThickness, Insulin, BMI = 0 se tratan como valores faltantes)
3. **Imputación**: `KNNImputer` (k=5), ajustado solo sobre el set de entrenamiento para evitar data leakage
4. **Escalado automatizado**: cada variable se escala con `StandardScaler` o `RobustScaler` según su coeficiente de asimetría (skew), aplicado dinámicamente mediante `ColumnTransformer`
5. **Modelado**: comparación de 3 algoritmos de clasificación
   - Logistic Regression
   - SVC (con `CalibratedClassifierCV` para probabilidades calibradas)
   - K-Nearest Neighbors
6. **Evaluación**: Accuracy, Precision, Recall, F1, ROC AUC — con foco en **Recall** dado el contexto médico (minimizar falsos negativos)
7. **Validación**: Cross Validation estratificada (5 folds) para confirmar la robustez de los resultados

## 📈 Resultados

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.734 | 0.603 | **0.704** | 0.650 | **0.814** |
| SVC | 0.747 | 0.653 | 0.593 | 0.621 | 0.810 |
| KNN | 0.701 | 0.577 | 0.556 | 0.566 | 0.734 |

**Confirmación con Cross Validation (5 folds):**

| Modelo | Recall promedio (CV) |
|---|---|
| **Logistic Regression** | 0.706 (± 0.040) |
| SVC | 0.603 (± 0.015) |
| KNN | 0.598 (± 0.059) |

## 🏆 Conclusión

Dado que el objetivo es la **detección de diabetes**, se priorizó **Recall** como métrica de referencia — un Falso Negativo (no detectar diabetes en un paciente que sí la tiene) es clínicamente más costoso que un Falso Positivo.

**Logistic Regression** resultó ser el modelo más adecuado: mejor Recall (70.4%), mejor ROC AUC (0.814), y resultado consistente entre el set de prueba y la validación cruzada — además de ser el más simple e interpretable de los tres.

## 🛠️ Tecnologías

- Python 3
- pandas, numpy
- scikit-learn
- seaborn, matplotlib

## 📁 Estructura

```
├── ML CLASIFICACION 1.0.2.ipynb    # Notebook completo con el análisis
└── README.md
```

## 🚀 Cómo ejecutar

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook "ML CLASIFICACION 1.0.2.ipynb"
```
