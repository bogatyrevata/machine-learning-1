crea un modelo relacional a partir de las tablas de éste link de kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce. Devolve el graficos del modelo

devolve las tablas q debo relaciona para responder a:

Caso 1: Predicción de Retrasos en la Entrega (Clasificación Binaria)

Objetivo: Predecir si un pedido sufrirá un retraso respecto a la fecha estimada por la plataforma (generar la columna is_delayed = 1, que no existe, si dias_retraso > 0; 0 en caso contrario).
Variables predictoras: Distancia cliente-vendedor, peso y volumen del paquete, estado geográfico de origen/destino, mes y día de la semana de la compra, valor del flete.
Algoritmos recomendados:
Baseline: Regresión Logística
Avanzados: Random Forest, XGBoost, LightGBM
Métricas de evaluación: ROC-AUC, PR-AUC y F1-Score (teniendo en cuenta el desbalance de clases si los retrasos representan un porcentaje minoritario).

el sql pasalo a python para obtener un dataframe