# Ejercicio Práctico - 21-10-26

**Dataset:** [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

Este documento contiene dos partes:

- **Parte I: Material de apoyo.** Análisis del dataset, consignas de procesamiento y limpieza, y propuestas de modelado.
- **Parte II: Consigna del Proyecto (entregable).** Predicción del tiempo real de entrega de pedidos.

---

# Parte I. Material de apoyo

## 1. Análisis del Dataset

El conjunto de datos contiene aproximadamente 100.000 órdenes de compra realizadas entre 2016 y 2018 en múltiples marketplaces de Brasil conectados por la plataforma Olist.

Se trata de un dataset relacional estructurado en 9 tablas (archivos CSV):

- `olist_customers_dataset.csv`: datos de los clientes (`customer_id`, `customer_unique_id`, ciudad, estado, prefijo de código postal).
- `olist_orders_dataset.csv`: estado del pedido y marcas de tiempo del ciclo de vida (purchase, approved, delivered_carrier, delivered_customer, estimated_delivery).
- `olist_order_items_dataset.csv`: detalle de productos por orden (`order_id`, `product_id`, `seller_id`, precio, valor del flete).
- `olist_order_payments_dataset.csv`: métodos de pago, número de cuotas (installments) y montos pagados.
- `olist_order_reviews_dataset.csv`: reseñas de los usuarios (`review_score` de 1 a 5, títulos, comentarios en portugués y fechas).
- `olist_products_dataset.csv`: atributos físicos (peso, dimensiones) y categorías de los productos.
- `olist_sellers_dataset.csv`: datos de los vendedores (ciudad, estado, código postal).
- `olist_geolocation_dataset.csv`: coordenadas (latitud/longitud) asociadas a los prefijos de códigos postales brasileños.
- `product_category_name_translation.csv`: traducción de las categorías de productos del portugués al inglés.

## 2. Consignas para el Procesamiento y Limpieza de Datos

### Consigna A: Consolidación y Unificación del Grafo Relacional

- **Criterio de agregación:** una orden (`order_id`) puede contener varios productos y métodos de pago. Antes de hacer un join, se deben agregar los ítems a nivel de orden (SUM de precios, SUM de fletes, COUNT de ítems) y pagos (SUM de valores, tipo de pago dominante).
- **Unión de tablas:** realizar joins entre `orders`, `customers`, `items_aggregated`, `payments_aggregated`, `products`, `sellers` y `reviews`.

### Consigna B: Tratamiento de Valores Faltantes (Missing Values)

- **Fechas de entrega:** filtrar o marcar por separado las órdenes con estado `canceled` o `unavailable`, las cuales no poseen `order_delivered_customer_date`.
- **Reseñas:** rellenar las variables textuales nulas (`review_comment_title`, `review_comment_message`) con identificadores como `"sin_comentario"`.
- **Categorías de productos:** asignar `"desconocido"` a los productos que no tengan categoría asignada.

### Consigna C: Ingeniería de Características (Feature Engineering)

**Métricas temporales:**

- Tiempo de entrega real = `order_delivered_customer_date` − `order_purchase_timestamp`
- Tiempo de entrega estimado = `order_estimated_delivery_date` − `order_purchase_timestamp`
- Días de retraso = `order_delivered_customer_date` − `order_estimated_delivery_date` (valores > 0 indican que la orden llegó tarde)

**Geolocalización y distancia:**

- Calcular la distancia (usando la fórmula de Haversine) entre las coordenadas del comprador (`customer_zip_code_prefix`) y del vendedor (`seller_zip_code_prefix`).

**Métricas físicas y financieras:**

- Calcular el volumen del producto: largo × ancho × alto.
- Calcular el ratio de flete sobre precio: `freight_value / price`.

### Consigna D: Codificación y Escalado

- **Codificación categórica:** aplicar One-Hot Encoding o Target Encoding para variables como `payment_type`, `customer_state`, `seller_state` y `product_category_name_english`.
- **Normalización / estandarización:** aplicar `StandardScaler` o `RobustScaler` sobre variables numéricas continuas (precios, pesos, distancias, días transcurridos) para evitar sesgos en algoritmos sensibles a la escala.

## 3. Consignas para Algoritmos de Predicción

A continuación se presentan 4 propuestas de modelado predictivo abordando distintas necesidades del negocio.

### Caso 1: Predicción de Retrasos en la Entrega (Clasificación Binaria)

- **Objetivo:** predecir si un pedido sufrirá un retraso respecto a la fecha estimada por la plataforma (generar la columna `is_delayed`, que no existe: 1 si `dias_retraso > 0`; 0 en caso contrario).
- **Variables predictoras:** distancia cliente-vendedor, peso y volumen del paquete, estado geográfico de origen/destino, mes y día de la semana de la compra, valor del flete.
- **Algoritmos recomendados:**
  - Baseline: Regresión Logística
  - Avanzados: Random Forest, XGBoost, LightGBM
- **Métricas de evaluación:** ROC-AUC, PR-AUC y F1-Score (teniendo en cuenta el desbalance de clases si los retrasos representan un porcentaje minoritario).

### Caso 2: Predicción de Calificación del Cliente / Satisfacción (Clasificación o NLP)

- **Objetivo:** predecir si la reseña del cliente será negativa (1 a 3 estrellas) o positiva (4 a 5 estrellas).
- **Variables predictoras:** días de retraso en la entrega, diferencia de precio vs. flete, número de ítems comprados, historial del vendedor.
- **Variante NLP (opcional):** aplicar TF-IDF o modelos de lenguaje (BERT/RoBERTa en portugués) sobre el texto de la reseña (`review_comment_message`) para predecir el `review_score`.
- **Algoritmos recomendados:** LightGBM, Random Forest, Logistic Regression / Naive Bayes (con TF-IDF).
- **Métricas de evaluación:** Accuracy, Weighted F1-Score.

### Caso 3: Predicción de Ventas y Demanda Temporal (Series Temporales / Regresión)

- **Objetivo:** predecir el volumen de ventas diarias/semanales agregadas por categoría de producto o estado.
- **Variables predictoras:** histórico de ventas, tendencia, variables de estacionalidad (mes, día del mes, día de la semana, festivos brasileños, fechas como Black Friday).
- **Algoritmos recomendados:**
  - Prophet de Meta
  - ARIMA / SARIMAX
  - XGBoost Regressor empleando variables de desfase (lags) y medias móviles (rolling statistics)
- **Métricas de evaluación:** MAE, RMSE, MAPE.

### Caso 4: Segmentación de Comportamiento de Clientes (Clustering No Supervisado)

- **Objetivo:** clasificar a los usuarios según la metodología RFM (Recency, Frequency, Monetary).
- **Variables predictoras:** recencia (días desde la última compra), frecuencia (total de compras asociadas al `customer_unique_id`), valor monetario (gasto acumulado).
- **Algoritmos recomendados:** K-Means, DBSCAN, Clustering Jerárquico.
- **Métricas de evaluación:** Coeficiente Silhouette, Índice Davies-Bouldin.

## 4. Estrategia de Validación Recomendada

**División temporal (time-based split):** al tratarse de transacciones ordenadas en el tiempo, se debe dividir el dataset de forma cronológica (por ejemplo, entrenar con datos de 2016 hasta mediados de 2018 y evaluar con los últimos meses de 2018) para evitar el data leakage (filtración de datos del futuro).

---

# Parte II. Consigna del Proyecto

# Predicción del Tiempo Real de Entrega de Pedidos (Lead Time Logístico)

## 1. Contexto del Negocio

Para una plataforma de comercio electrónico como Olist, estimar con precisión el tiempo de entrega es clave para gestionar la experiencia del usuario y optimizar la logística. Aunque la plataforma calcula una fecha de entrega estimada mediante reglas predefinidas, un modelo de Machine Learning predictivo permite estimar de forma continua y precisa el tiempo real de entrega en días desde el momento exacto en que se realiza la compra.

## 2. Objetivo Principal

Desarrollar, evaluar y comparar modelos de regresión supervisados para predecir la variable continua $y$, que representa el tiempo total de entrega (en días) transcurrido desde la fecha de compra hasta la entrega efectiva en el domicilio del cliente.

## 3. Definición de la Variable Objetivo ($y$)

A partir de la tabla `olist_orders_dataset.csv`, calcular la variable dependiente $y$:

$$y = \frac{\texttt{order\_delivered\_customer\_date} - \texttt{order\_purchase\_timestamp}}{24\ \text{horas}}$$

> **Nota:** se deben filtrar las órdenes no entregadas (`order_status != 'delivered'`) y registros con valores nulos en la fecha de entrega.

## 4. Construcción del Dataset de Entrenamiento ($X$)

El estudiante/analista deberá unificar las distintas tablas del dataset relacional para construir un conjunto de variables predictoras ($X$) a nivel de orden (`order_id`).

### A. Variables Geográficas y Logísticas

- **Distancia geográfica (km):** calcular la distancia de Haversine entre las coordenadas (`lat`, `lng`) del vendedor y del comprador usando `olist_geolocation_dataset.csv`.
- **Mismo estado:** variable binaria (1 si `customer_state == seller_state`, 0 en caso contrario).

### B. Variables del Producto e Ítems (tablas `olist_products_dataset.csv` y `olist_order_items_dataset.csv`)

- **Volumen total del paquete:** (largo × alto × ancho) / 1000 (en cm³ o dm³).
- **Peso total del paquete:** suma de `product_weight_g` por orden.
- **Cantidad de ítems:** número total de productos en la orden (`count_items`).
- **Diversidad de vendedores:** número de vendedores distintos involucrados en el mismo pedido.

### C. Variables Financieras (tablas `olist_order_items_dataset.csv` y `olist_order_payments_dataset.csv`)

- **Precio total de la orden:** suma del precio de los ítems.
- **Valor total del flete:** suma de `freight_value`.
- **Ratio de flete/precio:** $\dfrac{\texttt{freight\_value\_total}}{\texttt{price\_total}}$
- **Cuotas de pago:** número máximo de cuotas seleccionadas (`payment_installments`).

### D. Variables Temporales (extraídas de `order_purchase_timestamp`)

- Día de la semana de la compra (0 a 6).
- Hora del día (0 a 23).
- Mes de la compra (1 a 12).
- Binario de fin de semana / día laboral.

## 5. Requisitos de Procesamiento de Datos y Limpieza

### Tratamiento de Outliers

- Identificar y remover valores atípicos en la variable objetivo $y$ (por ejemplo, entregas con valores negativos o inconsistentes, o tiempos extremos superiores a 90-120 días).
- Tratar valores atípicos en dimensiones físicas o fletes utilizando métricas robustas (rango intercuartílico IQR o percentiles 1% y 99%).

### Ingeniería de Variables y Codificación

- Aplicar One-Hot Encoding o Target Encoding para variables categóricas con alta cardinalidad (por ejemplo, `customer_state`, `seller_state`, `product_category_name`).

### Escalado

- Transformar características continuas con distribuciones sesgadas mediante $\log(1 + x)$ o escaladores como `StandardScaler` / `RobustScaler`.

## 6. Estrategia de Validación del Modelo

**División temporal (time-based split):** para evitar la filtración temporal (data leakage), no utilizar una división aleatoria (`train_test_split` tradicional).

- **Entrenamiento:** datos comprendidos entre octubre de 2016 y mayo de 2018 (aprox. 80 % inicial de la línea de tiempo).
- **Prueba:** órdenes correspondientes al período junio de 2018 a agosto de 2018.

## 7. Modelos a Implementar y Comparar

Deberán entrenarse al menos tres algoritmos de distinta complejidad:

1. **Modelo baseline:** Regresión Lineal Simple / Ridge / Lasso.
2. **Modelo basado en árboles:** Random Forest Regressor.
3. **Modelo de boosting:** XGBoost Regressor / LightGBM Regressor / CatBoost.

## 8. Métricas de Evaluación Obligatorias

Evaluar el desempeño sobre el conjunto de prueba utilizando:

- **Error Absoluto Medio (MAE):**

$$MAE = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|$$

- **Raíz del Error Cuadrático Medio (RMSE):**

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}$$

- **Coeficiente de Determinación ($R^2$).**
- **Error Porcentual Absoluto Medio (MAPE).**

## 9. Entregables Esperados

1. **Código en Jupyter Notebook / Python:** limpio, documentado y reproducible.
2. **Gráfico de análisis de residuos:** distribución de errores ($y - \hat{y}$) para el mejor modelo.
3. **Importancia de características (feature importance):** identificar los 10 factores principales que más influyen en la demora de las entregas (por ejemplo, distancia geolocalizada, peso del producto, etc.).
4. **Conclusión de negocio:** comparación entre el tiempo de entrega predicho por el modelo y la estimación estática original que figuraba en la plataforma (`order_estimated_delivery_date`).
