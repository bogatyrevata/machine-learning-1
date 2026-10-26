# Árbol de Decisión / Дерево решений — Teoría / Теория

> Fuente / Источник: [python-course.eu — Decision Trees in Python](https://python-course.eu/machine-learning/decision-trees-in-python.php)

---

## 1. ¿Qué es? / Что это такое

**RU.** Дерево решений относится к обучению с учителем (*aprendizaje supervisado*): у нас есть данные, где ответ уже известен, и модель учится предсказывать его для новых данных. Модель задаёт цепочку вопросов вида «признак X больше порога?», на каждом шаге делит данные на группы и в конце даёт ответ. Дерево решает задачи **классификации** (ответ — класс: звезда/галактика, хороший/плохой кредит) и **регрессии** (ответ — число).

**ES.** El árbol de decisión es un algoritmo de aprendizaje supervisado que sirve para clasificación y regresión. Hace una serie de preguntas sobre las variables, divide los datos en grupos y en cada hoja entrega una predicción.

### Partes del árbol / Части дерева

```
        [raíz / корень]              ← primera pregunta
         /           \
   [nodo / узел]   [hoja / лист]     ← predicción
     /      \
 [hoja]   [hoja]
```

| Español | Русский | Significado / Что значит |
|---|---|---|
| Nodo raíz | Корневой узел | Primera pregunta, de aquí parte el árbol / Первый вопрос |
| Nodo interno | Внутренний узел | Pregunta intermedia / Промежуточный вопрос |
| Hoja (nodo hoja) | Лист | Final de la rama, contiene la predicción / Конец ветки, предсказание |
| Rama | Ветка | Conexión entre nodos (respuesta sí/no) |
| Variable descriptiva (feature) | Признак | Datos de entrada (p. ej. `redshift`) |
| Variable objetivo (target) | Целевая переменная | Lo que se predice (p. ej. `class`) |
| Profundidad | Глубина | Cantidad de preguntas seguidas desde la raíz hasta una hoja |

---

## 2. Idea principal / Главная идея

**RU.** На каждом шаге дерево ищет признак, который лучше всего разделяет данные так, чтобы в получившихся группах оказались почти одни и те же классы (группы стали «чистыми», *puras*). Так продолжается, пока не сработает условие остановки. Для нового объекта мы спускаемся от корня вниз по веткам до листа и берём ответ из него.

**ES.** La idea principal es encontrar la variable que mejor separa los datos, de modo que cada subgrupo sea lo más **puro** posible (casi todos de la misma clase). Se repite hasta cumplir un criterio de parada. Para predecir, se baja desde la raíz hasta una hoja.

**Ejemplo / Пример (из источника):** классификация животных — млекопитающее или рептилия. Признак `hair` (есть шерсть) делит данные идеально: шерсть есть → всегда млекопитающее. Значит, `hair` — самый информативный признак.

---

## 3. Medir la pureza: entropía y Gini / Измерение чистоты

**RU.** Нужна числовая мера того, насколько группа «перемешана».

| Español | Русский | Significado / Смысл |
|---|---|---|
| Entropía | Энтропия | 0 = grupo puro (один класс); чем больше, тем сильнее перемешано |
| Índice de Gini | Индекс Джини | Otra medida de impureza, más simple de calcular / Другая мера нечистоты |
| Ganancia de información | Прирост информации | Cuánto reduce la impureza una división / Насколько разделение уменьшило нечистоту |
| Impureza | Нечистота | Nombre general para entropía y Gini |

### Entropía (Shannon) / Энтропия

$$H = -\sum_k p_k \cdot \log_2(p_k)$$

donde $p_k$ es la proporción de la clase $k$ en el grupo / где $p_k$ — доля класса $k$ в группе.

**Ejemplo / Пример:** 6 mamíferos y 4 reptiles de 10:

$$H = -(0.6\cdot\log_2 0.6 + 0.4\cdot\log_2 0.4) = 0.971$$

Если бы все 10 были одного класса, было бы $H = 0$.

### Gini

$$Gini = 1 - \sum_k p_k^2$$

Para los mismos datos / Для тех же данных: $1 - (0.6^2 + 0.4^2) = 0.48$.

### Ganancia de información / Прирост информации

Entropía antes de dividir menos la entropía ponderada después / Энтропия до разделения минус взвешенная энтропия после:

$$IG = H(D) - \sum_t \frac{|D_t|}{|D|}\cdot H(D_t)$$

**Ejemplo / Пример:** para la variable `legs`, IG = 0.971 − 0.414 = **0.557**, más que `toothed` (0.007) y `breathes` (0.144). Por eso `legs` se convierte en la raíz.

> En `scikit-learn` el parámetro `criterion` acepta `"gini"` (por defecto), `"entropy"` o `"log_loss"`. Los resultados suelen ser muy parecidos.

---

## 4. Algoritmo de construcción (ID3) / Алгоритм построения

**Pasos / Шаги:**

1. Tomar todos los datos de entrenamiento / Взять все обучающие данные.
2. Calcular la ganancia de información de cada variable / Посчитать прирост информации для каждого признака.
3. Elegir la variable con mayor ganancia: ese es el nodo / Выбрать признак с максимальным приростом.
4. Dividir los datos según los valores de esa variable / Разделить данные по значениям признака.
5. Repetir 2–4 en cada parte (**recursión**) / Повторить для каждой части.
6. Parar según un criterio y crear una hoja / Остановиться и создать лист.

### Criterios de parada / Когда останавливаемся

1. Todos los objetos del grupo son de la misma clase (grupo puro) / Группа чистая.
2. Ya no quedan variables / Признаки закончились.
3. Ya no quedan datos en el grupo / Данных не осталось.

Además se suelen fijar límites a mano / Дополнительно задают ограничения:

- `max_depth`: profundidad máxima / максимальная глубина
- `min_samples_split`: mínimo de objetos para dividir un nodo / минимум объектов для разделения узла
- `min_samples_leaf`: mínimo de objetos en una hoja / минимум объектов в листе

**ES.** ID3 (Quinlan, 1986) elige la variable con mayor ganancia, divide y repite recursivamente. Otras versiones: C4.5, C5.0 y **CART** (la que usa `scikit-learn`).

**Variables continuas / Непрерывные признаки** (nuestro caso: `redshift`, `u`, `g`, ...): el árbol prueba umbrales del tipo «variable ≤ 0.5?» y elige el mejor. La misma variable puede usarse varias veces con distintos umbrales / Один признак может использоваться несколько раз с разными порогами.

---

## 5. Ventajas y desventajas / Плюсы и минусы

| Ventajas / Плюсы | Desventajas / Минусы |
|---|---|
| Fácil de interpretar («caja blanca») / Легко интерпретировать | Propenso al **sobreajuste** / Склонно к переобучению |
| No necesita normalizar variables / Не нужна нормализация | Pequeños cambios en los datos pueden cambiar mucho el árbol / Нестабильность |
| Maneja datos numéricos y categóricos / Числа и категории | Clases desbalanceadas sesgan el resultado hacia la clase frecuente / Несбалансированные классы |
| Modela relaciones no lineales / Нелинейные зависимости | Divide el espacio en «rectángulos» / Делит пространство на прямоугольники |
| Captura interacciones entre variables / Взаимодействия признаков | Variables con muchos niveles tienen ventaja / Признаки с многими уровнями |

---

## 6. Sobreajuste y cómo evitarlo / Переобучение

**RU.** Без ограничений дерево растёт, пока не запомнит каждый обучающий пример. На train оно показывает почти 100%, а на новых данных результат хуже. Это и есть *sobreajuste* (overfitting).

**Cómo combatirlo / Как бороться:**

- Limitar la profundidad (`max_depth`) y el tamaño de nodos (`min_samples_leaf`) / Ограничить глубину и размер узлов.
- **Poda (pruning)**: recortar hojas si la precisión no baja / Подрезка листьев, если точность не падает.
- Métodos de conjunto / Ансамбли: *bagging*, *boosting*, **Random Forest** (muchos árboles votan / много деревьев голосуют).

**Conexión con nuestra tarea / Связь с заданием:** por eso usamos **validación cruzada** y comparamos las métricas de train y test. Если на train результат намного выше, чем на test, дерево переобучилось.

---

## 7. Evaluación / Оценка модели

| Español | Русский | Significado |
|---|---|---|
| Validación cruzada (cross-validation) | Кросс-валидация | Dividir los datos en *k* partes; entrenar con *k−1* y validar con la restante, *k* veces |
| Exactitud (accuracy) | Точность по всем | Proporción de predicciones correctas |
| Precisión (precision) | Точность по классу | De lo predicho como clase X, cuánto es realmente X |
| Sensibilidad (recall) | Полнота | De los X reales, cuántos encontró el modelo |
| F1-score | F1-мера | Media armónica de precisión y recall |
| Matriz de confusión | Матрица ошибок | Tabla: predicho vs. real, por clase |

---

## 8. Árbol en scikit-learn (5 pasos) / Дерево в sklearn

1. Elegir el modelo: `DecisionTreeClassifier`.
2. Definir hiperparámetros (`max_depth`, `criterion`, ...).
3. Preparar `X` (variables) e `y` (objetivo).
4. Entrenar: `.fit(X_train, y_train)`.
5. Predecir: `.predict(X_test)`.

---

## 9. Resumen / Шпаргалка

- El árbol divide los datos con preguntas, eligiendo la división con mayor **ganancia de información** (o menor Gini).
- Grupo **puro** → entropía 0 → hoja.
- Principal desventaja: **sobreajuste**; se reduce con límites de profundidad, poda y ensambles.
- No requiere normalización de datos.
- Para evaluar: **validación cruzada** + **métricas** + **matriz de confusión**.

---

## Referencias / Источники

- Tobias Schlagenhauf, *Decision Trees in Python*, python-course.eu (2022).
- Mitchell, T. (1997). *Machine Learning*. McGraw-Hill.
- Kelleher, Mac Namee, D'Arcy (2015). *Machine Learning for Predictive Data Analytics*. MIT Press.