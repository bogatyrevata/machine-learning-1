# Conclusiones / Выводы — Árbol de Decisión

Modelo: `DecisionTreeClassifier` (scikit-learn), división 80 % entrenamiento / 20 % prueba (estratificada), validación cruzada estratificada de 5 folds.

---

## 1. Credit Score (`train.csv`, 100 000 filas)

### Español

**Datos.** Variable objetivo `Credit_Score` con tres clases: Standard (≈ 53 % de la muestra de prueba), Poor (≈ 29 %) y Good (≈ 18 %); las clases están desbalanceadas. Se eliminaron columnas identificadoras y de texto libre (`ID`, `Customer_ID`, `Month`, `Name`, `SSN`, `Type_of_Loan`, `Credit_History_Age`), se convirtieron a número los valores guardados como texto (por ejemplo `23_`), se filtraron edades inválidas, se imputaron los faltantes con la mediana y se codificaron las variables categóricas con `get_dummies`.

**Método.** División 80 % / 20 % estratificada; búsqueda en rejilla (`GridSearchCV`) sobre el conjunto de entrenamiento con validación cruzada estratificada de 5 folds, probando `criterion` (gini, entropy, log_loss) y `max_depth` (None, 2 a 10).

**Resultados.**

| Métrica | Valor |
|---|---|
| Mejores parámetros | `criterion='gini'`, `max_depth=10` |
| Accuracy por fold (CV) | 0.7031, 0.7085, 0.7047, 0.7043, 0.7045 |
| Accuracy media (CV) | 0.7050 ± 0.0018 |
| Accuracy en entrenamiento | 0.7276 |
| Accuracy en prueba | 0.7154 |

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| Good | 0.58 | 0.67 | 0.62 | 3459 |
| Poor | 0.75 | 0.68 | 0.71 | 5641 |
| Standard | 0.75 | 0.75 | 0.75 | 10345 |

Matriz de confusión (filas = real, columnas = predicho):

| | Good | Poor | Standard |
|---|---|---|---|
| **Good** | 2304 | 63 | 1092 |
| **Poor** | 370 | 3827 | 1444 |
| **Standard** | 1327 | 1238 | 7780 |

**Conclusiones.**

1. **Estabilidad.** Los cinco folds varían poco (0.7031–0.7085) y la desviación estándar es 0.0018.
2. **Sobreajuste leve.** La accuracy de entrenamiento (0.7276) es mayor que la de prueba (0.7154); la diferencia (≈ 0.012) es pequeña pero mayor que en Stellar (≈ 0.001). La accuracy de validación (0.7050) y la de prueba (0.7154) son coherentes entre sí.
3. **Mejora frente a la línea base.** Responder siempre "Standard" daría ≈ 53 % de accuracy; el modelo alcanza ≈ 71.5 %, una mejora clara pero un desempeño moderado.
4. **Elección de parámetros.** Con `gini`, la accuracy de validación sigue subiendo con la profundidad (0.6993 con 8, 0.7030 con 9 y 0.7050 con 10), y el mejor valor está en el **borde de la rejilla**, por lo que podría mejorar con profundidades mayores (12, 15, 20); no se probó. Con `entropy` y `log_loss` (resultados idénticos entre sí) el máximo está en profundidad 8 (0.7011) y luego baja. La accuracy de entrenamiento crece más rápido que la de validación al aumentar la profundidad.
5. **Errores.** De 5534 errores en prueba, 5101 (≈ 92 %) involucran a la clase intermedia Standard (Good↔Standard y Poor↔Standard). Good↔Poor son solo 433 casos (≈ 8 %): el modelo casi no confunde los extremos. La clase más débil es Good (precision 0.58): de los clientes clasificados como Good, solo el 58 % lo es realmente; 1092 clientes Good fueron clasificados como Standard.
6. **Variables importantes.** `Outstanding_Debt` (0.346), `Credit_Mix_Standard` (0.200) e `Interest_Rate` (0.093) suman ≈ 64 % de la importancia; siguen `Payment_of_Min_Amount_Yes` (0.058), `Changed_Credit_Limit` (0.051) y `Num_Credit_Card` (0.048). Las variables dummy de `Credit_Mix` (`Standard` y `_`) suman ≈ 0.25. La categoría `_` de `Credit_Mix` representa un dato faltante en el origen y el modelo la usa como información.
7. **Comparación con una configuración inicial** (`max_depth=8`, `gini`): la accuracy media de validación pasó de 0.6993 a 0.7050 y la de prueba de 0.71 a 0.7154; el recall de Good bajó de 0.70 a 0.67 y su precision subió de 0.56 a 0.58.

**Limitaciones.** Cada cliente aparece en 8 filas (una por mes) y algunas variables, como `Outstanding_Debt`, son casi constantes para un mismo cliente. Con una división aleatoria, filas del mismo cliente pueden quedar en entrenamiento y prueba, y el árbol podría "reconocer" clientes en lugar de aprender reglas generales; esto podría explicar en parte por qué la accuracy sigue subiendo con la profundidad y hace que la accuracy sea algo optimista para clientes nuevos (hipótesis, no verificada). Una división por cliente (`GroupShuffleSplit`) permitiría comprobarlo.

### Русский

**Данные.** Целевая переменная `Credit_Score`: Standard (≈ 53 % тестовой выборки), Poor (≈ 29 %), Good (≈ 18 %); классы несбалансированы. Удалены идентификаторы и текстовые столбцы (`ID`, `Customer_ID`, `Month`, `Name`, `SSN`, `Type_of_Loan`, `Credit_History_Age`), числа, записанные как текст (например, `23_`), приведены к числовому виду, убраны некорректные значения возраста, пропуски заполнены медианой, категории преобразованы через `get_dummies`.

**Метод.** Разделение 80 / 20 со стратификацией; перебор параметров (`GridSearchCV`) на обучающей части с 5-фолдовой стратифицированной кросс-валидацией: `criterion` (gini, entropy, log_loss) и `max_depth` (None, 2–10).

**Результаты.** Лучшие параметры: `criterion='gini'`, `max_depth=10`. Accuracy на кросс-валидации 0.7050 ± 0.0018, на обучении 0.7276, на тесте 0.7154. Таблицы по классам и матрица ошибок приведены выше.

**Выводы.**

1. **Стабильность.** Пять фолдов различаются слабо (0.7031–0.7085), стандартное отклонение 0.0018.
2. **Небольшое переобучение.** Accuracy на обучении (0.7276) выше, чем на тесте (0.7154): разница ≈ 0.012, небольшая, но больше, чем у Stellar (≈ 0.001). Accuracy на валидации (0.7050) и на тесте (0.7154) согласуются.
3. **Сравнение с базовым уровнем.** Если всегда отвечать «Standard», accuracy ≈ 53 %. Модель даёт ≈ 71.5 %: заметное улучшение, но качество умеренное.
4. **Выбор параметров.** При `gini` accuracy на валидации продолжает расти с глубиной (0.6993 при 8, 0.7030 при 9, 0.7050 при 10), и лучшее значение находится **на краю сетки**, поэтому большие глубины (12, 15, 20) могли бы дать лучше; это не проверялось. У `entropy` и `log_loss` (результаты у них одинаковые) максимум при глубине 8 (0.7011), дальше качество падает. Accuracy на обучении растёт с глубиной быстрее, чем на валидации.
5. **Ошибки.** Из 5534 ошибок на тесте 5101 (≈ 92 %) связаны со средним классом Standard (Good↔Standard и Poor↔Standard). Good↔Poor всего 433 случая (≈ 8 %): крайние классы почти не путаются. Самый слабый класс — Good (precision 0.58): из клиентов, названных моделью Good, лишь 58 % действительно Good; 1092 клиента Good отнесены к Standard.
6. **Важные признаки.** `Outstanding_Debt` (0.346), `Credit_Mix_Standard` (0.200) и `Interest_Rate` (0.093) дают вместе ≈ 64 % важности; далее `Payment_of_Min_Amount_Yes` (0.058), `Changed_Credit_Limit` (0.051), `Num_Credit_Card` (0.048). Dummy-признаки `Credit_Mix` (`Standard` и `_`) вместе ≈ 0.25. Категория `_` у `Credit_Mix` обозначает пропуск в исходных данных, и модель использует её как информацию.
7. **Сравнение с первой конфигурацией** (`max_depth=8`, `gini`): средняя CV-accuracy выросла с 0.6993 до 0.7050, на тесте с 0.71 до 0.7154; recall у Good снизился с 0.70 до 0.67, а precision вырос с 0.56 до 0.58.

**Ограничения.** Каждый клиент встречается в 8 строках (по одной на месяц), а некоторые признаки, например `Outstanding_Debt`, почти постоянны для одного клиента. При случайном разделении строки одного клиента могут попасть и в train, и в test, и дерево может «узнавать» клиентов, а не учить общие правила; возможно, этим отчасти объясняется, что accuracy растёт с глубиной, и оценка оптимистична для новых клиентов (гипотеза, не проверена). Проверить можно разделением по клиентам (`GroupShuffleSplit`).

---

## 2. Stellar Classification (`star_classification.csv`, 100 000 filas)

### Español

**Datos.** Variable objetivo `class` con tres clases: GALAXY (≈ 59 % de la muestra de prueba), QSO (≈ 19 %) y STAR (≈ 22 %). Se eliminó la fila con valor atípico (-9999) en `u`, `g`, `z` y las columnas identificadoras (`obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `spec_obj_ID`, `plate`, `MJD`, `fiber_ID`). Quedan 8 variables: `alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift`.

**Método.** División 80 % / 20 % estratificada; búsqueda en rejilla (`GridSearchCV`) sobre el conjunto de entrenamiento con validación cruzada estratificada de 5 folds, probando `criterion` (gini, entropy, log_loss) y `max_depth` (None, 2 a 10).

**Resultados.**

| Métrica | Valor |
|---|---|
| Mejores parámetros | `criterion='entropy'`, `max_depth=8` |
| Accuracy por fold (CV) | 0.9730, 0.9751, 0.9726, 0.9723, 0.9749 |
| Accuracy media (CV) | 0.9736 ± 0.0012 |
| Accuracy en entrenamiento | 0.9777 |
| Accuracy en prueba | 0.9764 |

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| GALAXY | 0.97 | 0.99 | 0.98 | 11889 |
| QSO | 0.96 | 0.92 | 0.94 | 3792 |
| STAR | 1.00 | 1.00 | 1.00 | 4319 |

Matriz de confusión (filas = real, columnas = predicho):

| | GALAXY | QSO | STAR |
|---|---|---|---|
| **GALAXY** | 11722 | 150 | 17 |
| **QSO** | 302 | 3490 | 0 |
| **STAR** | 4 | 0 | 4315 |

**Conclusiones.**

1. **Estabilidad y sobreajuste.** Los cinco folds varían muy poco (0.9723–0.9751) y la accuracy de entrenamiento (0.9777) es casi igual a la de prueba (0.9764). La diferencia de 0.0013 indica que no hay sobreajuste relevante.
2. **Mejora frente a una línea base.** Responder siempre "GALAXY" daría ≈ 59 % de accuracy; el modelo alcanza ≈ 97.6 %.
3. **Elección de parámetros.** `entropy` y `log_loss` dan resultados idénticos (en scikit-learn calculan el mismo criterio). Las profundidades 7 a 10 obtienen una accuracy de validación casi igual (0.9733–0.9736); las diferencias son menores que la desviación estándar entre folds (≈ 0.001), por lo que no son significativas. A medida que crece la profundidad, la accuracy de entrenamiento sigue subiendo (de 0.9762 con profundidad 7 a 0.9828 con 10) mientras la de validación se mantiene: el árbol empieza a memorizar sin ganar capacidad de generalización. Se eligió `max_depth=8`, el mejor valor de la búsqueda.
4. **Errores.** De los 473 errores en prueba, 452 (≈ 96 %) son confusiones entre GALAXY y QSO: 302 quasares clasificados como galaxia (recall de QSO 0.92) y 150 galaxias clasificadas como quasar. La clase STAR casi no se confunde (21 errores en total, ninguno con QSO).
5. **Variables importantes.** `redshift` concentra el 91.2 % de la importancia, seguida de `g` (5.2 %), `u` (1.5 %) y `z` (1.2 %). Las coordenadas `alpha` y `delta` aportan menos de 0.1 % cada una. Esto es coherente con que el corrimiento al rojo separa muy bien los tipos de objeto (las estrellas de nuestra galaxia tienen `redshift` cercano a 0).
6. **Comparación con una configuración inicial** (`max_depth=6`, `gini`): la accuracy media de validación pasó de 0.9666 a 0.9736 y el recall de QSO de 0.88 a 0.92.

### Русский

**Данные.** Целевая переменная `class`: GALAXY (≈ 59 % теста), QSO (≈ 19 %), STAR (≈ 22 %). Удалена строка с выбросом -9999 в `u`, `g`, `z` и столбцы-идентификаторы. Остаётся 8 признаков: `alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift`.

**Метод.** Разделение 80 / 20 со стратификацией; перебор параметров (`GridSearchCV`) на обучающей части с 5-фолдовой стратифицированной кросс-валидацией: `criterion` (gini, entropy, log_loss) и `max_depth` (None, 2–10).

**Результаты.** Лучшие параметры: `criterion='entropy'`, `max_depth=8`. Accuracy на кросс-валидации 0.9736 ± 0.0012, на обучении 0.9777, на тесте 0.9764. Таблицы по классам и матрица ошибок приведены выше.

**Выводы.**

1. **Стабильность и переобучение.** Пять фолдов различаются слабо (0.9723–0.9751), accuracy на обучении (0.9777) почти равна accuracy на тесте (0.9764), разница 0.0013. Существенного переобучения нет.
2. **Сравнение с базовым уровнем.** Если всегда отвечать «GALAXY», accuracy ≈ 59 %. Модель даёт ≈ 97.6 %.
3. **Выбор параметров.** `entropy` и `log_loss` дают одинаковые результаты (в scikit-learn это один и тот же критерий). Глубины 7–10 почти не различаются по качеству на валидации (0.9733–0.9736), а разница меньше разброса между фолдами (≈ 0.001), то есть статистически незначима. При росте глубины точность на обучении продолжает расти (0.9762 → 0.9828), а на валидации стоит на месте: дерево начинает запоминать данные, не улучшая обобщение. Выбрана `max_depth=8`, лучшая в поиске.
4. **Ошибки.** Из 473 ошибок на тесте 452 (≈ 96 %) — путаница между GALAXY и QSO: 302 квазара названы галактиками (recall QSO 0.92) и 150 галактик названы квазарами. STAR почти не путается (21 ошибка, ни одной с QSO).
5. **Важность признаков.** На `redshift` приходится 91.2 % важности, далее `g` (5.2 %), `u` (1.5 %), `z` (1.2 %). Координаты `alpha` и `delta` дают менее 0.1 % каждая. Это согласуется с тем, что красное смещение хорошо разделяет типы объектов (звёзды нашей Галактики имеют `redshift` около 0).
6. **Сравнение с первой конфигурацией** (`max_depth=6`, `gini`): средняя CV-accuracy выросла с 0.9666 до 0.9736, recall QSO с 0.88 до 0.92.

---

## 3. Comparación general / Общее сравнение

### Español

| | Stellar | Credit Score |
|---|---|---|
| Mejores parámetros | `entropy`, `max_depth=8` | `gini`, `max_depth=10` |
| Accuracy media (CV) | 0.9736 ± 0.0012 | 0.7050 ± 0.0018 |
| Accuracy en prueba | 0.9764 | 0.7154 |
| Diferencia entrenamiento − prueba | ≈ 0.001 | ≈ 0.012 |
| Clase más difícil | QSO (recall 0.92) | Good (precision 0.58) |
| Variable más importante | `redshift` (0.91) | `Outstanding_Debt` (0.35) |

- **Por qué el desempeño difiere.** En Stellar una sola variable (`redshift`) concentra el 91 % de la importancia y separa muy bien las clases; casi todos los errores (≈ 96 %) están entre GALAXY y QSO. En Credit Score la importancia se reparte entre varias variables (deuda, mezcla de créditos, tasa de interés, pagos mínimos) y las clases son ordenadas y se solapan: el 92 % de los errores involucra a la clase intermedia Standard, mientras que Good↔Poor casi no se confunden.
- **Sobreajuste.** Es despreciable en Stellar y leve en Credit Score (mayor diferencia entre entrenamiento y prueba).
- **Ventajas del árbol observadas.** Es fácil de interpretar (importancia de variables y errores por clase), no requiere normalizar los datos y la validación cruzada permitió comprobar la estabilidad de los resultados.
- **Mejoras posibles.** Ampliar la rejilla de `max_depth` en Credit Score (el máximo está en el borde); ajustar `min_samples_leaf`; usar `class_weight="balanced"` para mejorar la clase Good; dividir por cliente (`GroupShuffleSplit`); probar `RandomForestClassifier`.

### Русский

| | Stellar | Credit Score |
|---|---|---|
| Лучшие параметры | `entropy`, `max_depth=8` | `gini`, `max_depth=10` |
| Средняя accuracy (CV) | 0.9736 ± 0.0012 | 0.7050 ± 0.0018 |
| Accuracy на тесте | 0.9764 | 0.7154 |
| Разница train − test | ≈ 0.001 | ≈ 0.012 |
| Самый трудный класс | QSO (recall 0.92) | Good (precision 0.58) |
| Главный признак | `redshift` (0.91) | `Outstanding_Debt` (0.35) |

- **Почему качество различается.** В Stellar один признак (`redshift`) даёт 91 % важности и хорошо разделяет классы; почти все ошибки (≈ 96 %) приходятся на пару GALAXY↔QSO. В Credit Score важность распределена между несколькими признаками (долг, структура кредитов, процентная ставка, минимальные платежи), а классы упорядочены и перекрываются: 92 % ошибок связаны со средним классом Standard, а Good↔Poor почти не путаются.
- **Переобучение.** В Stellar пренебрежимо мало, в Credit Score небольшое (разница между train и test больше).
- **Плюсы дерева, которые проявились.** Легко интерпретировать (важность признаков, ошибки по классам), не нужна нормализация, кросс-валидация показала стабильность результатов.
- **Возможные улучшения.** Расширить сетку `max_depth` в Credit Score (максимум на краю); подобрать `min_samples_leaf`; использовать `class_weight="balanced"` для класса Good; разделять данные по клиентам (`GroupShuffleSplit`); попробовать `RandomForestClassifier`.