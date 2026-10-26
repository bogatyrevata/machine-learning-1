# common/evaluacion.py
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.tree import DecisionTreeClassifier

# Rejilla de parámetros (la misma idea que usó el profesor en clase)
PARAM_GRID = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}


def revisar_datos(df):
    """Revisión inicial: dimensiones, nulos, duplicados, tipos y estadísticas."""
    print("Dimensiones:", df.shape)
    print("Total de valores nulos:", df.isnull().sum().sum())
    print("Filas completamente duplicadas:", df.duplicated().sum())
    print("\n--- Tipos de datos ---")
    df.info()
    print("\n--- Estadísticas descriptivas ---")
    display(df.describe(include="all").T)


def analizar(X, y, param_grid=PARAM_GRID, guardar_matriz=None):
    """Árbol de decisión con búsqueda de parámetros, validación cruzada,
    métricas y matriz de confusión. Devuelve el mejor modelo."""
    # 1. División 80 % entrenamiento / 20 % prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Validación cruzada estratificada de 5 folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Búsqueda de los mejores parámetros (solo con datos de entrenamiento)
    busqueda = GridSearchCV(
        DecisionTreeClassifier(random_state=42), param_grid,
        cv=cv, scoring="accuracy", n_jobs=-1, return_train_score=True)
    busqueda.fit(X_train, y_train)
    modelo = busqueda.best_estimator_

    print("Mejores parámetros:", busqueda.best_params_)
    print(f"Mejor accuracy en validación cruzada: {busqueda.best_score_:.4f}\n")

    resultados = pd.DataFrame(busqueda.cv_results_)
    print("--- Mejores 10 combinaciones evaluadas ---")
    display(resultados[["param_criterion", "param_max_depth", "mean_train_score",
                        "mean_test_score", "std_test_score"]]
            .sort_values("mean_test_score", ascending=False).head(10))

    # 4. Accuracy por fold del mejor modelo
    scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="accuracy")
    print("Accuracy por fold:", scores.round(4))
    print(f"Media: {scores.mean():.4f} ± {scores.std():.4f}")

    # 5. Comparación entrenamiento vs. prueba (control de sobreajuste)
    print(f"\nAccuracy en entrenamiento: {modelo.score(X_train, y_train):.4f}")
    print(f"Accuracy en prueba:        {modelo.score(X_test, y_test):.4f}\n")

    # 6. Métricas por clase
    y_pred = modelo.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 7. Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=modelo.classes_).plot(cmap="Blues", ax=ax)
    ax.set_title("Matriz de confusión (conjunto de prueba)")
    if guardar_matriz:
        ruta = Path(guardar_matriz)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.show()

    # 8. Importancia de las variables
    importancias = pd.Series(modelo.feature_importances_,
                             index=X.columns).sort_values(ascending=False)
    print("Top 10 variables más importantes:")
    print(importancias.head(10).round(4))

    return modelo