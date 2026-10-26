# Árbol de Decisión: Stellar Classification y Credit Score

Trabajo práctico de la Tecnicatura en Ciencia de Datos e Inteligencia Artificial.
Se analizan dos conjuntos de datos de Kaggle con un **árbol de decisión** (`DecisionTreeClassifier`), aplicando **validación cruzada**, calculando **métricas de clasificación** y generando la **matriz de confusión**.

## Conjuntos de datos

| # | Conjunto | Objetivo | Clases | Filas |
|---|---|---|---|---|
| 1 | [Stellar Classification Dataset (SDSS17)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) | `class` | GALAXY, STAR, QSO | 100 000 |
| 2 | [Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) | `Credit_Score` | Good, Standard, Poor | 100 000 |

## Metodología

1. **Preparación de datos** (en cada notebook).
   - Stellar: se elimina la fila con valores atípicos (-9999) en `u`, `g`, `z` y las columnas identificadoras.
   - Credit Score: se eliminan columnas identificadoras, se convierten a número los valores guardados como texto (por ejemplo `23_`), se filtran edades inválidas, se imputan faltantes con la mediana y se codifican las variables categóricas con `get_dummies`.
2. **División** 80 % entrenamiento / 20 % prueba, estratificada (`train_test_split`, `random_state=42`).
3. **Validación cruzada** estratificada de 5 folds (`StratifiedKFold`) sobre el conjunto de entrenamiento.
4. **Entrenamiento** del árbol (`max_depth` limitado para evitar sobreajuste) y **evaluación** sobre el conjunto de prueba: `classification_report` y **matriz de confusión**.

Los pasos 2 a 4 están implementados en la función `analizar` de [`src/evaluacion.py`](src/evaluacion.py), que usan ambos notebooks.

## Resultados

| Conjunto | `max_depth` | Accuracy CV (media ± desv.) | Accuracy en prueba |
|---|---|---|---|
| Stellar Classification | 6 | 0.9666 ± 0.0018 | 0.97 |
| Credit Score | 8 | 0.6993 ± 0.0021 | 0.71 |

Las conclusiones completas (en español y ruso), con las métricas por clase, están en [`reports/conclusiones.md`](reports/conclusiones.md) y al final de cada notebook.

## Estructura del repositorio

```
machine-learning-1/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md                      # origen de los datos
│   ├── star_classification.csv
│   └── credit-score-classifier/
│       └── train.csv
├── notebooks/
│   ├── 01_stellar_classification.ipynb
│   └── 02_credit_score.ipynb
├── src/
│   ├── __init__.py
│   └── evaluacion.py                  # función analizar()
├── reports/
│   └── conclusiones.md
└── teoria/
    └── Arbol_de_decision.md           # teoría del árbol de decisión
```

## Cómo ejecutar

Requiere Python 3.11 o superior (desarrollado con Python 3.14).

```bash
# 1. Crear y activar el entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Obtener los datos (ver data/README.md) y abrir los notebooks
jupyter lab
```

Ejecutar los notebooks desde la carpeta `notebooks/` (el kernel debe tener como directorio de trabajo `notebooks/`, porque importan `src.evaluacion` y leen los datos desde `../data`).

### Datos

Los archivos esperados son:

- `data/star_classification.csv`
- `data/credit-score-classifier/train.csv`

Si no están en el repositorio, descargarlos desde los enlaces de la tabla de conjuntos de datos (botón **Download** de Kaggle, o con `kagglehub`) y colocarlos en esas rutas. El archivo `test.csv` de Credit Score **no se utiliza**: no contiene la columna objetivo, por lo que no permite calcular métricas. La evaluación se hace con la partición 80/20 de `train.csv`.

## Limitaciones

- En Credit Score cada cliente aparece en 8 filas (una por mes). Con una división aleatoria, filas de un mismo cliente pueden quedar en entrenamiento y prueba, por lo que la accuracy podría ser algo optimista para clientes completamente nuevos.
- La profundidad máxima del árbol se fijó a mano (6 y 8); no se realizó búsqueda de hiperparámetros.

## Tecnologías

Python, pandas, NumPy, scikit-learn, matplotlib, Jupyter. Versiones fijadas en `requirements.txt`.

---

## Кратко по-русски

Задание: анализ двух наборов данных Kaggle (Stellar Classification и Credit Score) деревом решений с кросс-валидацией (5 фолдов), метриками и матрицей ошибок. Общая функция `analizar` лежит в `src/evaluacion.py`, ноутбуки в `notebooks/`, выводы на двух языках в `reports/conclusiones.md`, теория в `teoria/`.

Результаты: Stellar — accuracy 0.97 (CV 0.9666), Credit Score — accuracy 0.71 (CV 0.6993). Запуск: создать `.venv`, `pip install -r requirements.txt`, положить данные в `data/` (см. `data/README.md`), запускать ноутбуки из папки `notebooks/`.