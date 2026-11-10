# Caso 1: Predicción de retrasos en la entrega (Olist)

## 1. Planteamiento del problema

**Dataset:** Brazilian E-Commerce Public Dataset by Olist, 9 tablas relacionadas (CSV) con pedidos de una tienda online de Brasil.

**Objetivo:** predecir si un pedido llegará más tarde de la fecha que prometió la plataforma.

**Tipo de problema:** clasificación binaria, es decir, el modelo tiene dos respuestas posibles: 0 o 1.

**Variable objetivo** (target). No existe en el dataset, la creamos nosotros:

```
dias_retraso = fecha_entrega_real − fecha_entrega_estimada   (en días)
is_delayed   = 1, si dias_retraso > 0
               0, en caso contrario
```

**Variables predictoras** (features): lo que el modelo usará para predecir.

| Variable | De dónde sale |
|---|---|
| distancia cliente-vendedor | coordenadas de `geolocation` |
| peso y volumen del paquete | `products` |
| estado de origen y de destino | `sellers`, `customers` |
| mes y día de la semana de la compra | `orders` |
| valor del flete | `order_items` |

---

## 2. Modelo relacional de Olist

Un **modelo relacional** es una forma de guardar datos en varias tablas conectadas mediante claves. Así los datos no se duplican: la información de un cliente se guarda una sola vez y los pedidos hacen referencia a él.

### 2.1. Claves

- **Clave primaria** (PK): columna (o conjunto de columnas) que identifica de forma única cada fila de una tabla. Por ejemplo, `order_id` en `orders`.
- **Clave foránea** (FK): columna que hace referencia a la clave primaria de otra tabla. Por ejemplo, `customer_id` en `orders` apunta a `customers`.
- **Clave compuesta**: clave primaria formada por varias columnas. Por ejemplo, en `order_items` la clave es `(order_id, order_item_id)`: un pedido puede tener varios ítems.

### 2.2. Tablas y relaciones

| Tabla | PK | Qué guarda |
|---|---|---|
| `customers` | `customer_id` | clientes, código postal, ciudad, estado |
| `orders` | `order_id` | pedidos, estado y todas las fechas |
| `order_items` | `(order_id, order_item_id)` | ítems del pedido, precio, flete |
| `order_payments` | `(order_id, payment_sequential)` | pagos |
| `order_reviews` | `(review_id, order_id)` | reseñas |
| `products` | `product_id` | categoría, peso, dimensiones |
| `sellers` | `seller_id` | vendedores, código postal, estado |
| `geolocation` | no tiene | coordenadas por código postal |
| `product_category_name_translation` | `product_category_name` | traducción de categorías al inglés |

**Relaciones (cardinalidad):**

- `customers` 1 → N `orders`: un cliente puede hacer muchos pedidos.
- `orders` 1 → N `order_items`, `order_payments`, `order_reviews`.
- `products` 1 → N `order_items` y `sellers` 1 → N `order_items`. La tabla `order_items` es una **tabla puente** entre pedidos, productos y vendedores.
- `geolocation` se relaciona con `customers` y `sellers` solo de forma **lógica** (por código postal). No es una FK real: en `geolocation` hay muchas filas por código postal y no existe una clave única.

---

## 3. Tablas necesarias para el Caso 1

Se necesitan 6 de las 9 tablas.

| Tabla | Columnas | Para qué |
|---|---|---|
| `orders` | fechas, `order_status`, `customer_id` | variable objetivo, mes, día de la semana |
| `customers` | `customer_zip_code_prefix`, `customer_state` | punto de destino |
| `order_items` | `product_id`, `seller_id`, `freight_value` | enlace pedido-producto-vendedor, flete |
| `products` | peso y dimensiones | peso y volumen |
| `sellers` | `seller_zip_code_prefix`, `seller_state` | punto de origen |
| `geolocation` | `lat`, `lng` | distancia |

**No se necesitan:** `order_payments`, `order_reviews`, `product_category_name_translation`.

Camino de los joins:

```
orders ──► customers ──► geolocation (destino)
   │
   └──► order_items ──► products
              │
              └──► sellers ──► geolocation (origen)
```

---

## 4. Teoría: cómo armar el dataset

### 4.1. JOIN (unión de tablas)

Un JOIN pega filas de dos tablas según el valor de una clave en común.

- **INNER JOIN** (`how="inner"`) conserva solo las filas que tienen pareja en ambas tablas.
- **LEFT JOIN** (`how="left"`) conserva todas las filas de la tabla izquierda. Si no hay pareja, las columnas de la tabla derecha quedan en `NaN` (valor faltante).

En nuestro caso usamos inner para las relaciones principales (pedido → cliente) y left para los datos opcionales (peso del producto, coordenadas). Así no perdemos pedidos en silencio: vemos los faltantes y decidimos qué hacer con ellos.

### 4.2. Granularidad

La **granularidad** indica qué representa una fila de la tabla. Para el modelo necesitamos **una fila = un pedido**. Pero en `order_items` un pedido puede tener varias filas. Por eso, antes de la unión final **agregamos** (groupby + agg):

| Variable | Cómo se agrega |
|---|---|
| flete | suma |
| peso | suma |
| volumen | suma |
| distancia | promedio de los ítems |

Si no se hace, al unir las tablas las filas del pedido se **multiplican**, y un mismo pedido aparece varias veces en los datos. Eso también distorsiona las métricas.

### 4.3. Agregar `geolocation`

En `geolocation` hay muchas filas por código postal, con coordenadas ligeramente distintas. Si se une directamente, los pedidos se multiplican. Por eso primero agrupamos por código postal y calculamos el **promedio** de latitud y longitud, para tener una sola fila por código.

### 4.4. Distancia: fórmula de Haversine

La Tierra es redonda, así que con latitud y longitud no se puede calcular la distancia como en un plano. La fórmula de Haversine da la distancia sobre el arco de círculo máximo entre dos puntos:

$$a = \sin^2\!\left(\frac{\Delta\varphi}{2}\right) + \cos\varphi_1 \cdot \cos\varphi_2 \cdot \sin^2\!\left(\frac{\Delta\lambda}{2}\right)$$

$$d = 2R \cdot \arcsin\!\left(\sqrt{a}\right)$$

Donde $\varphi$ es la latitud, $\lambda$ la longitud (en radianes) y $R = 6371$ km el radio de la Tierra.

### 4.5. Fuga de datos (data leakage)

Hay **fuga de datos** cuando entre las variables predictoras entra información que no existiría en el momento de predecir. El modelo "espía" la respuesta, las métricas en test salen irrealmente buenas y en la vida real el modelo no funciona.

En nuestro problema la predicción se hace **en el momento de la compra**. Por lo tanto, **no** se pueden incluir como predictoras:

- `order_delivered_customer_date` (fecha real de entrega);
- `dias_retraso` (es la propia respuesta);
- las fechas de aprobación y de envío al transportista;
- las reseñas (`order_reviews`), porque se escriben después de la entrega.

### 4.6. Filtrado de pedidos

Nos quedamos solo con los pedidos con estado `delivered` y con fecha real de entrega no nula. Si no, no se puede calcular el retraso: los pedidos cancelados o aún no entregados no tienen respuesta correcta.

---

## 5. Teoría: desbalance de clases y métricas

El **desbalance de clases** ocurre cuando una clase aparece mucho más que la otra. En los retrasos, probablemente haya muchos menos pedidos con `is_delayed = 1` que pedidos entregados a tiempo.

Por eso la **exactitud** (accuracy) engaña. Si los retrasos son el 8 %, un modelo que siempre responde 0 tendría 92 % de accuracy, pero no sirve para nada.

Para evaluar usamos otras métricas. Primero, las definiciones:

- **TP** (verdadero positivo): predijimos retraso y lo hubo.
- **FP** (falso positivo): predijimos retraso y no lo hubo.
- **FN** (falso negativo): no detectamos un retraso real.

| Métrica | Qué muestra |
|---|---|
| **Precisión** (precision) = TP / (TP + FP) | de todos los retrasos predichos, qué proporción es real |
| **Exhaustividad** (recall) = TP / (TP + FN) | qué proporción de los retrasos reales encontramos |
| **F1-Score** | media armónica de precision y recall: $F1 = 2 \cdot \frac{P \cdot R}{P + R}$ |
| **ROC-AUC** | qué tan bien el modelo separa la clase 1 de la clase 0 con cualquier umbral (1.0 es perfecto, 0.5 es azar) |
| **PR-AUC** | área bajo la curva precision-recall; con desbalance es más informativa que ROC-AUC porque mira solo la clase minoritaria |

**Umbral:** el modelo devuelve una probabilidad de retraso entre 0 y 1, y la decisión "1 o 0" se toma con un umbral (por defecto 0.5). ROC-AUC y PR-AUC no dependen de un único umbral: evalúan la calidad de las probabilidades en conjunto.

---

## 6. Glosario (Español → Русский)

| Español | Русский |
|---|---|
| clasificación binaria | бинарная классификация |
| variable objetivo | целевая переменная |
| variables predictoras | признаки (предикторы) |
| modelo relacional | реляционная модель |
| clave primaria (PK) | первичный ключ |
| clave foránea (FK) | внешний ключ |
| clave compuesta | составной ключ |
| cardinalidad | кардинальность (тип связи) |
| relación 1 a N | связь «один ко многим» |
| tabla puente | связующая таблица |
| integridad referencial | ссылочная целостность |
| unión (join) | соединение таблиц |
| granularidad | гранулярность |
| agregar | агрегировать |
| valores nulos / faltantes | пропущенные значения |
| imputar | заполнять пропуски |
| fuga de datos | утечка данных |
| desbalance de clases | дисбаланс классов |
| umbral | порог |
| precisión (precision) | точность |
| exhaustividad (recall) | полнота |
| exactitud (accuracy) | accuracy (доля верных ответов) |

---

## 7. Resumen

1. De las 9 tablas, el Caso 1 necesita 6: `orders`, `customers`, `order_items`, `products`, `sellers`, `geolocation`.
2. La variable objetivo `is_delayed` se construye con dos fechas de `orders`.
3. Los datos se llevan a "una fila = un pedido" mediante agregación.
4. La distancia se calcula con la fórmula de Haversine, con coordenadas de `geolocation` ya agregada.
5. Se excluyen de las predictoras todas las variables desconocidas al momento de la compra (fuga de datos).
6. Por el posible desbalance de clases, los modelos se evalúan con ROC-AUC, PR-AUC y F1, y no con accuracy.



# Caso 1: предсказание задержек доставки (Olist)

## 1. Постановка задачи

**Датасет:** Brazilian E-Commerce Public Dataset by Olist, 9 связанных таблиц (CSV) с заказами интернет-магазина в Бразилии.

**Задача:** предсказать, будет ли заказ доставлен позже, чем обещала платформа.

**Тип задачи:** бинарная классификация (clasificación binaria), то есть у модели два возможных ответа: 0 или 1.

**Целевая переменная** (variable objetivo, target). В датасете её нет, мы создаём её сами:

```
dias_retraso = fecha_entrega_real − fecha_entrega_estimada   (в днях)
is_delayed   = 1, если dias_retraso > 0
               0, в противном случае
```

**Признаки** (variables predictoras, features) — то, по чему модель будет предсказывать:

| Признак | Откуда берём |
|---|---|
| расстояние клиент – продавец | координаты из `geolocation` |
| вес и объём посылки | `products` |
| штат отправления и назначения | `sellers`, `customers` |
| месяц и день недели покупки | `orders` |
| стоимость доставки (flete) | `order_items` |

---

## 2. Реляционная модель Olist

**Реляционная модель** — способ хранить данные в нескольких таблицах, связанных между собой через ключи. Так данные не дублируются: информация о клиенте хранится один раз, а заказы ссылаются на него.

### 2.1. Ключи

- **Первичный ключ** (clave primaria, PK) — столбец (или набор столбцов), который однозначно определяет строку таблицы. Например, `order_id` в `orders`.
- **Внешний ключ** (clave foránea, FK) — столбец, который ссылается на первичный ключ другой таблицы. Например, `customer_id` в `orders` ссылается на `customers`.
- **Составной ключ** (clave compuesta) — первичный ключ из нескольких столбцов. Например, в `order_items` ключ `(order_id, order_item_id)`: в одном заказе может быть несколько позиций.

### 2.2. Таблицы и связи

| Таблица | PK | Что хранит |
|---|---|---|
| `customers` | `customer_id` | клиенты, почтовый индекс, город, штат |
| `orders` | `order_id` | заказы, статус и все даты |
| `order_items` | `(order_id, order_item_id)` | позиции заказа, цена, flete |
| `order_payments` | `(order_id, payment_sequential)` | платежи |
| `order_reviews` | `(review_id, order_id)` | отзывы |
| `products` | `product_id` | категория, вес, размеры |
| `sellers` | `seller_id` | продавцы, почтовый индекс, штат |
| `geolocation` | нет | координаты по почтовым индексам |
| `product_category_name_translation` | `product_category_name` | перевод категорий на английский |

**Связи (cardinalidad):**

- `customers` 1 → N `orders`: у клиента может быть много заказов.
- `orders` 1 → N `order_items`, `order_payments`, `order_reviews`.
- `products` 1 → N `order_items` и `sellers` 1 → N `order_items`. Таблица `order_items` является **связующей** (tabla puente) между заказами, товарами и продавцами.
- `geolocation` связана с `customers` и `sellers` только **логически** (по почтовому индексу). Это не настоящий FK: в `geolocation` на один индекс приходится много строк, и уникального ключа нет.

---

## 3. Какие таблицы нужны для Caso 1

Нужны 6 из 9 таблиц.

| Таблица | Столбцы | Зачем |
|---|---|---|
| `orders` | даты, `order_status`, `customer_id` | целевая переменная, месяц, день недели |
| `customers` | `customer_zip_code_prefix`, `customer_state` | точка назначения |
| `order_items` | `product_id`, `seller_id`, `freight_value` | связь заказа с товаром и продавцом, flete |
| `products` | вес и размеры | вес и объём |
| `sellers` | `seller_zip_code_prefix`, `seller_state` | точка отправления |
| `geolocation` | `lat`, `lng` | расстояние |

**Не нужны:** `order_payments`, `order_reviews`, `product_category_name_translation`.

Путь соединений:

```
orders ──► customers ──► geolocation (назначение)
   │
   └──► order_items ──► products
              │
              └──► sellers ──► geolocation (отправление)
```

---

## 4. Теория: как собрать датасет

### 4.1. JOIN (соединение таблиц)

JOIN склеивает строки двух таблиц по совпадающему значению ключа.

- **INNER JOIN** (`how="inner"`) оставляет только строки, у которых нашлась пара в обеих таблицах.
- **LEFT JOIN** (`how="left"`) оставляет все строки левой таблицы. Если пары нет, в столбцах правой таблицы будет `NaN` (пропуск).

В нашем случае для основных связей (заказ → клиент) используем inner, а для необязательных данных (вес товара, координаты) используем left. Тогда мы не теряем заказы молча, а видим пропуски и решаем, что с ними делать.

### 4.2. Гранулярность (granularidad)

**Гранулярность** — что именно представляет одна строка таблицы. Для модели нам нужна **одна строка = один заказ**. Но в `order_items` у одного заказа может быть несколько строк. Поэтому перед финальным соединением мы **агрегируем** (groupby + agg):

| Признак | Как агрегируем |
|---|---|
| flete | сумма |
| вес | сумма |
| объём | сумма |
| расстояние | среднее по позициям |

Если этого не сделать, при соединении строки заказа **размножатся**, и один заказ попадёт в данные несколько раз. Это тоже искажает метрики.

### 4.3. Агрегирование `geolocation`

В `geolocation` для одного индекса много строк с немного разными координатами. Если соединить напрямую, заказы размножатся. Поэтому сначала группируем по индексу и берём **среднюю** широту и долготу, получая одну строку на индекс.

### 4.4. Расстояние: формула гаверсинусов (Haversine)

Земля круглая, поэтому по широте и долготе нельзя считать расстояние как на плоскости. Формула Хаверсина даёт расстояние по дуге большого круга между двумя точками:

$$a = \sin^2\!\left(\frac{\Delta\varphi}{2}\right) + \cos\varphi_1 \cdot \cos\varphi_2 \cdot \sin^2\!\left(\frac{\Delta\lambda}{2}\right)$$

$$d = 2R \cdot \arcsin\!\left(\sqrt{a}\right)$$

Здесь $\varphi$ — широта, $\lambda$ — долгота (в радианах), $R = 6371$ км — радиус Земли.

### 4.5. Утечка данных (fuga de datos, data leakage)

**Утечка данных** — это когда в признаки попадает информация, которой не было бы в момент предсказания. Модель «подглядывает» в ответ, и метрики на тесте получаются нереально хорошими, а в реальной жизни модель не работает.

Для нашей задачи прогноз делается **в момент покупки**. Значит, в признаки **нельзя** включать:

- `order_delivered_customer_date` (реальная дата доставки);
- `dias_retraso` (это и есть ответ);
- даты одобрения и передачи перевозчику;
- отзывы (`order_reviews`), они пишутся после доставки.

### 4.6. Фильтрация заказов

Берём только заказы со статусом `delivered` и непустой реальной датой доставки. Иначе нельзя посчитать задержку: у отменённых и ещё не доставленных заказов правильного ответа нет.

---

## 5. Теория: дисбаланс классов и метрики

**Дисбаланс классов** (desbalance de clases) — когда один класс встречается намного чаще другого. Для задержек, скорее всего, заказов с `is_delayed = 1` значительно меньше, чем доставленных вовремя.

Из-за этого обычная **точность** (accuracy) вводит в заблуждение. Если задержек 8%, модель «всегда говорит 0» получит accuracy 92%, но она бесполезна.

Для оценки используем другие метрики. Сначала определим понятия:

- **TP** (true positive): предсказали задержку, и она была.
- **FP** (false positive): предсказали задержку, а её не было.
- **FN** (false negative): задержку не заметили.

| Метрика | Что показывает |
|---|---|
| **Precision** (точность) = TP / (TP + FP) | из всех предсказанных задержек, какая доля настоящая |
| **Recall** (полнота) = TP / (TP + FN) | какую долю настоящих задержек мы нашли |
| **F1-Score** | среднее гармоническое precision и recall: $F1 = 2 \cdot \frac{P \cdot R}{P + R}$ |
| **ROC-AUC** | насколько хорошо модель отделяет класс 1 от класса 0 при любом пороге (1.0 — идеально, 0.5 — случайно) |
| **PR-AUC** | площадь под кривой precision–recall; при дисбалансе информативнее ROC-AUC, потому что смотрит только на редкий класс |

**Порог** (umbral): модель выдаёт вероятность задержки от 0 до 1, а решение «1 или 0» мы принимаем по порогу (по умолчанию 0.5). ROC-AUC и PR-AUC не зависят от одного порога, они оценивают качество вероятностей в целом.

---

## 6. Словарь терминов (Español → Русский)

| Español | Русский |
|---|---|
| clasificación binaria | бинарная классификация |
| variable objetivo | целевая переменная |
| variables predictoras | признаки (предикторы) |
| modelo relacional | реляционная модель |
| clave primaria (PK) | первичный ключ |
| clave foránea (FK) | внешний ключ |
| clave compuesta | составной ключ |
| cardinalidad | кардинальность (тип связи) |
| relación 1 a N | связь «один ко многим» |
| tabla puente | связующая таблица |
| integridad referencial | ссылочная целостность |
| unión (join) | соединение таблиц |
| granularidad | гранулярность |
| agregar | агрегировать |
| valores nulos | пропущенные значения |
| imputar | заполнять пропуски |
| fuga de datos | утечка данных |
| desbalance de clases | дисбаланс классов |
| umbral | порог |
| precisión / exhaustividad | точность (precision) / полнота (recall) |

---

## 7. Итог

1. Из 9 таблиц для Caso 1 нужны 6: `orders`, `customers`, `order_items`, `products`, `sellers`, `geolocation`.
2. Целевую переменную `is_delayed` строим из двух дат в `orders`.
3. Данные приводим к виду «одна строка = один заказ» через агрегирование.
4. Расстояние считаем по формуле Хаверсина, а координаты берём из агрегированной `geolocation`.
5. Из признаков исключаем всё, что неизвестно на момент покупки (утечка данных).
6. Из-за возможного дисбаланса классов оцениваем модели по ROC-AUC, PR-AUC и F1, а не по accuracy.