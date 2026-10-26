# Árbol de Decisión: Stellar Classification y Credit Score

Trabajo práctico de la Tecnicatura en Ciencia de Datos e Inteligencia Artificial.
Se analizan dos conjuntos de datos de Kaggle con un **árbol de decisión** (`DecisionTreeClassifier`): búsqueda de parámetros con **validación cruzada** (`GridSearchCV`), **métricas de clasificación** y **matriz de confusión**.

## Conjuntos de datos

| # | Conjunto | Objetivo | Clases | Filas |
|---|---|---|---|---|
| 1 | [Stellar Classification Dataset (SDSS17)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) | `class` | GALAXY, STAR, QSO | 100 000 |
| 2 | [Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) | `Credit_Score` | Good, Standard, Poor | 100 000 |

## Metodología

1. **Revisión de datos** (`revisar_datos`): dimensiones, valores nulos, duplicados, tipos y estadísticas descriptivas.
2. **Preparación de datos** (en cada notebook).
   - Stellar: se elimina la fila con valores atípicos (-9999) en `u`, `g`, `z` y las columnas identificadoras.
   - Credit Score: se eliminan columnas identificadoras y de texto libre, se convierten a número los valores guardados como texto (por ejemplo `23_`), se filtran edades inválidas, se imputan faltantes con la mediana y se codifican las variables categóricas con `get_dummies`.
3. **División** 80 % entrenamiento / 20 % prueba, estratificada (`random_state=42`).
4. **Búsqueda de parámetros** con `GridSearchCV` sobre el conjunto de entrenamiento y **validación cruzada estratificada de 5 folds** (`StratifiedKFold`): `criterion` ∈ {gini, entropy, log_loss} y `max_depth` ∈ {None, 2, …, 10}.
5. **Evaluación** del mejor modelo sobre el conjunto de prueba: comparación entrenamiento vs. prueba, `classification_report`, **matriz de confusión** e importancia de variables.

Los pasos 3 a 5 están implementados en la función `analizar` de [`../common/evaluacion.py`](../common/evaluacion.py), que usan ambos notebooks.

## Resultados

| Conjunto | Mejores parámetros | Accuracy CV (media ± desv.) | Accuracy entrenamiento | Accuracy prueba |
|---|---|---|---|---|
| Stellar Classification | `entropy`, `max_depth=8` | 0.9736 ± 0.0012 | 0.9777 | 0.9764 |
| Credit Score | `gini`, `max_depth=10` | 0.7050 ± 0.0018 | 0.7276 | 0.7154 |

Las conclusiones completas (en español y ruso), con las métricas por clase y las matrices de confusión, están en [`conclusiones.md`](conclusiones.md) y al final de cada notebook. La teoría del árbol de decisión está en [`teoria.md`](teoria.md).

## Estructura del repositorio

```
machine-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── common/
│   ├── __init__.py
│   └── evaluacion.py                  # funciones analizar() y revisar_datos()
└── arbol-decision/
    ├── README.md                      # este archivo
    ├── conclusiones.md
    ├── teoria.md
    ├── data/
    │   ├── README.md                  # origen de los datos
    │   ├── star_classification.csv
    │   └── credit-score-classifier/
    │       └── train.csv
    ├── 01_stellar_classification.ipynb
    └── 02_credit_score.ipynb
```

## Cómo ejecutar

Requiere Python 3.11 o superior (desarrollado con Python 3.14).

```bash
# 1. Crear y activar el entorno virtual (en la raíz del repositorio)
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Abrir los notebooks de la carpeta arbol-decision/
jupyter lab
```

Los notebooks deben ejecutarse con `arbol-decision/` como directorio de trabajo: importan `common.evaluacion` (mediante `sys.path.append("..")`) y leen los datos desde `data/`.

### Datos

Los archivos utilizados son:

- `data/star_classification.csv`
- `data/credit-score-classifier/train.csv`

Si no estuvieran presentes, descargarlos desde los enlaces de la tabla de conjuntos de datos (botón **Download** de Kaggle, o con `kagglehub`) y colocarlos en esas rutas. El archivo `test.csv` de Credit Score **no se utiliza**: no contiene la columna objetivo, por lo que no permite calcular métricas. La evaluación se hace con la partición 80/20 de `train.csv`.

## Limitaciones

- En Credit Score cada cliente aparece en 8 filas (una por mes), y algunas variables son casi constantes por cliente. Con una división aleatoria, filas de un mismo cliente pueden quedar en entrenamiento y prueba, por lo que la accuracy podría ser algo optimista para clientes nuevos. Una división por cliente (`GroupShuffleSplit`) permitiría comprobarlo.
- En Credit Score la mejor profundidad (10) está en el borde de la rejilla; podrían probarse profundidades mayores.
- No se ajustaron otros hiperparámetros (`min_samples_leaf`, `class_weight`) ni se probaron modelos de conjunto.

## Tecnologías

Python, pandas, NumPy, scikit-learn, matplotlib, Jupyter. Versiones fijadas en `requirements.txt`.

---

## Кратко по-русски

Задание: анализ двух наборов данных Kaggle (Stellar Classification и Credit Score) деревом решений с подбором параметров через `GridSearchCV` (кросс-валидация на 5 фолдов), метриками и матрицей ошибок. Общие функции `analizar` и `revisar_datos` лежат в `common/evaluacion.py`, ноутбуки и данные в `arbol-decision/`, выводы на двух языках в `conclusiones.md`, теория в `teoria.md`.

Результаты: Stellar — лучшие параметры `entropy`, `max_depth=8`, accuracy на тесте 0.9764 (CV 0.9736); Credit Score — `gini`, `max_depth=10`, accuracy на тесте 0.7154 (CV 0.7050). Запуск: создать `.venv` в корне, `pip install -r requirements.txt`, запускать ноутбуки из папки `arbol-decision/`.