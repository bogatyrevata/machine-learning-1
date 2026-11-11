# Unidad 6: Evaluación y Validación / Блок 6: Оценка и валидация

🇪🇸 [Español](#-español) · 🇷🇺 [Русский](#-русский)

**Notebook:** `Evaluacion_Validacion.ipynb`

---

# 🇪🇸 Español

## 1. Overfitting vs Underfitting y compromiso sesgo-varianza

- **Sesgo (Bias):** desviación entre la predicción esperada del modelo y el valor real. Sesgo alto = modelo demasiado simple → **underfitting**: error alto en train y en validación.
- **Varianza (Variance):** sensibilidad del modelo a pequeñas variaciones del conjunto de entrenamiento. Varianza alta = el modelo memoriza el ruido → **overfitting**: $E_{train} \ll E_{val}$.

Descomposición del error cuadrático medio:

$$E\left[(y-\hat f(x))^2\right] = \text{Bias}[\hat f(x)]^2 + \text{Var}[\hat f(x)] + \sigma^2$$

donde $\sigma^2$ es el **error irreducible** (ruido propio de los datos).

| Problema | Estrategias |
|---|---|
| Underfitting | Más complejidad (grado polinómico, profundidad del árbol), *feature engineering*, reducir la regularización ($\lambda \to 0$) |
| Overfitting | Regularización (L1, L2, ElasticNet, Dropout), más datos, reducción de dimensionalidad, *early stopping*, ensambles (Bagging, Random Forest) |

> **Nota práctica:** `bias_variance_decomp` de `mlxtend` calcula el sesgo contra `y_test` (con ruido), por lo que el valor incluye $\sigma^2$.

## 2. Train/Test split y fuga de datos (Data Leakage)

- **Holdout:** Train (70–80 %), Validation opcional (10–15 %) para hiperparámetros y Test (10–20 %) aislado hasta la evaluación final.
- **Data leakage:** información del conjunto de evaluación contamina el entrenamiento o el preprocesamiento. Ejemplo típico: `StandardScaler` o `SelectKBest` ajustados con **todo** el dataset antes del split.
- **Solución:** `fit` solo con train, `transform` sobre test; en la práctica, encapsular todo en un `Pipeline`.
- **Muestreo estratificado** (`stratify=y`): mantiene la proporción de clases en train y test; clave en datasets desbalanceados.

## 3. Validación cruzada y búsqueda de hiperparámetros

- **K-Fold:** K particiones; se entrena K veces con K−1 folds y se valida con 1; se promedian las métricas (reduce la varianza de la estimación).
- **Stratified K-Fold:** conserva la proporción de clases en cada fold (estándar en clasificación).
- **TimeSeriesSplit:** para series temporales (datos no i.i.d.); respeta el orden cronológico. `KFold(shuffle=True)` provoca fuga temporal.
- **GridSearchCV:** búsqueda exhaustiva en una grilla; costosa si el espacio es grande.
- **RandomizedSearchCV:** muestrea `n_iter` combinaciones; mejor uso del cómputo en espacios de alta dimensión.

## 4. Métricas con datos desbalanceados: ROC-AUC vs PR-AUC

$$TPR = \frac{TP}{TP+FN},\quad FPR = \frac{FP}{FP+TN},\quad Precision = \frac{TP}{TP+FP},\quad Recall = \frac{TP}{TP+FN}$$

Con muchos negativos, $FP+TN$ es enorme y la FPR casi no cambia aunque aumenten los falsos positivos → el ROC-AUC resulta **optimista**. La curva Precision-Recall no usa los TN, por lo que los falsos positivos penalizan directamente la precisión.

| Criterio | ROC-AUC | PR-AUC / Average Precision |
|---|---|---|
| Ejes | TPR vs FPR | Precision vs Recall |
| Sensibilidad al desbalance | Baja | Alta |
| Baseline (azar) | 0.50 | Proporción de la clase positiva $\frac{P}{P+N}$ |
| Uso recomendado | Clases balanceadas | Fraude, fallas, diagnósticos raros, spam |

## 5. Curvas de aprendizaje (Learning Curves)

- **Alta varianza:** brecha grande y persistente entre train (≈1.0) y validación → más datos, modelo más simple o más regularización.
- **Alto sesgo:** ambas curvas convergen pronto a un valor bajo → más complejidad o nuevas variables.

## 6. Actividades del notebook

| # | Actividad |
|---|---|
| 1 | Reemplazar `sin()` por otra función verdadera |
| 2 | `polyfit` + `bias_variance_decomp()` para evaluar sesgo y varianza |
| 3 | Mejorar `DecisionTreeRegressor` (profundidad, hojas mínimas, Bagging/RF) |
| 4 | Validación cruzada: GridSearchCV vs RandomizedSearchCV |
| 5 | Ejercicios Unidad 6: Holdout vs Stratified K-Fold, data leakage, Grid vs Random con XGBoost |

---

# 🇷🇺 Русский

## 1. Переобучение и недообучение, компромисс смещения и дисперсии

- **Смещение (Bias):** отклонение ожидаемого предсказания модели от истинного значения. Высокое смещение = слишком простая модель → **недообучение (underfitting)**: высокая ошибка и на обучающей, и на валидационной выборке.
- **Дисперсия (Variance):** чувствительность модели к небольшим изменениям обучающей выборки. Высокая дисперсия = модель запоминает шум → **переобучение (overfitting)**: $E_{train} \ll E_{val}$.

Разложение среднеквадратичной ошибки:

$$E\left[(y-\hat f(x))^2\right] = \text{Bias}[\hat f(x)]^2 + \text{Var}[\hat f(x)] + \sigma^2$$

где $\sigma^2$ — **неустранимая ошибка** (шум, присущий данным).

| Проблема | Способы борьбы |
|---|---|
| Недообучение | Усложнить модель (степень полинома, глубина дерева), *feature engineering*, ослабить регуляризацию ($\lambda \to 0$) |
| Переобучение | Регуляризация (L1, L2, ElasticNet, Dropout), больше данных, снижение размерности, ранняя остановка, ансамбли (Bagging, Random Forest) |

> **Практическое замечание:** `bias_variance_decomp` из `mlxtend` считает смещение относительно `y_test` (с шумом), поэтому значение включает $\sigma^2$.

## 2. Разбиение train/test и утечка данных (Data Leakage)

- **Holdout:** Train (70–80 %), необязательная Validation (10–15 %) для подбора гиперпараметров и Test (10–20 %), который не трогают до финальной оценки.
- **Утечка данных:** информация из тестовой/валидационной выборки попадает в обучение или предобработку. Типичный пример: `StandardScaler` или `SelectKBest`, обученные на **всём** датасете до разбиения.
- **Решение:** `fit` только на train, `transform` на test; на практике — всё упаковать в `Pipeline`.
- **Стратифицированное разбиение** (`stratify=y`): сохраняет пропорции классов в train и test; важно для несбалансированных данных.

## 3. Кросс-валидация и подбор гиперпараметров

- **K-Fold:** K частей; модель обучается K раз на K−1 частях и проверяется на одной; метрики усредняются (оценка получается менее шумной).
- **Stratified K-Fold:** сохраняет пропорцию классов в каждой части (стандарт для классификации).
- **TimeSeriesSplit:** для временных рядов (данные не i.i.d.); сохраняет хронологию. `KFold(shuffle=True)` приводит к временной утечке.
- **GridSearchCV:** полный перебор по сетке; дорого при большом пространстве поиска.
- **RandomizedSearchCV:** случайно выбирает `n_iter` комбинаций; эффективнее по ресурсам в многомерных пространствах.

## 4. Метрики при несбалансированных классах: ROC-AUC и PR-AUC

$$TPR = \frac{TP}{TP+FN},\quad FPR = \frac{FP}{FP+TN},\quad Precision = \frac{TP}{TP+FP},\quad Recall = \frac{TP}{TP+FN}$$

При огромном числе отрицательных примеров $FP+TN$ очень велико, и FPR почти не меняется даже при росте числа ложноположительных → ROC-AUC получается **завышенным**. Кривая Precision-Recall не использует TN, поэтому ложноположительные напрямую снижают precision.

| Критерий | ROC-AUC | PR-AUC / Average Precision |
|---|---|---|
| Оси | TPR и FPR | Precision и Recall |
| Чувствительность к дисбалансу | Низкая | Высокая |
| Baseline (случайная модель) | 0.50 | Доля положительного класса $\frac{P}{P+N}$ |
| Когда использовать | Сбалансированные классы | Фрод, поломки, редкие диагнозы, спам |

## 5. Кривые обучения (Learning Curves)

- **Высокая дисперсия:** большой устойчивый разрыв между train (≈1.0) и валидацией → больше данных, упростить модель или усилить регуляризацию.
- **Высокое смещение:** обе кривые рано сходятся к низкому значению → усложнить модель или добавить признаки.

## 6. Задания ноутбука

| № | Задание |
|---|---|
| 1 | Заменить `sin()` на другую истинную функцию |
| 2 | `polyfit` + `bias_variance_decomp()` для оценки смещения и дисперсии |
| 3 | Улучшить `DecisionTreeRegressor` (глубина, минимум в листе, Bagging/RF) |
| 4 | Кросс-валидация: GridSearchCV и RandomizedSearchCV |
| 5 | Упражнения Unidad 6: Holdout и Stratified K-Fold, утечка данных, Grid и Random с XGBoost |