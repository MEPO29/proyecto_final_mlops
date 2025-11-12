# -*- coding: utf-8 -*-
"""
Fase 4 - Modelado con MLflow 
incluye: carga de splits, experimentos con varios algoritmos, logging y registro de modelos.
"""

import os
import argparse
import warnings
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# Utilidades
def ensure_dense(X):
    """Convierte a denso si es sparse"""
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            return X.toarray()
    except Exception:
        pass
    return X

def save_fig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

def eval_and_log_artifacts(y_true, y_proba, y_pred, out_dir):
    """Calcula métricas, guarda y loggea artefactos gráficos."""
    os.makedirs(out_dir, exist_ok=True)

    # Métricas base
    metrics = {}
    metrics["accuracy"]  = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred))
    metrics["recall"]    = float(recall_score(y_true, y_pred))
    metrics["f1"]        = float(f1_score(y_true, y_pred))
    try:
        metrics["roc_auc"]  = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["roc_auc"] = float("nan")

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha="center", va="center")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    save_fig(cm_path)
    mlflow.log_artifact(cm_path)

    # Curva ROC
    try:
        fig = plt.figure()
        RocCurveDisplay.from_predictions(y_true, y_proba)
        plt.title("Curva ROC")
        roc_path = os.path.join(out_dir, "roc_curve.png")
        save_fig(roc_path)
        mlflow.log_artifact(roc_path)
    except Exception:
        pass

    # Curva Precision-Recall
    try:
        fig = plt.figure()
        PrecisionRecallDisplay.from_predictions(y_true, y_proba)
        plt.title("Curva Precision-Recall")
        pr_path = os.path.join(out_dir, "precision_recall_curve.png")
        save_fig(pr_path)
        mlflow.log_artifact(pr_path)
    except Exception:
        pass

    return metrics

def load_processed(processed_dir):
    """Carga splits y preprocessor desde la Fase 3."""
    splits = joblib.load(os.path.join(processed_dir, "splits.pkl"))
    preproc = joblib.load(os.path.join(processed_dir, "preprocessor.pkl"))
    X_train, y_train, X_valid, y_valid, X_test, y_test = splits
    return X_train, y_train, X_valid, y_valid, X_test, y_test, preproc

# Definición de modelos y espacios de hiperparámetros
def get_search_spaces(random_state=42):
    spaces = {}

    # 1) Logistic Regression
    spaces["LogisticRegression"] = {
        "estimator": LogisticRegression(random_state=random_state, max_iter=200, n_jobs=-1, solver="saga"),
        "params": {
            "C": np.logspace(-3, 2, 20),
            "penalty": ["l1", "l2", "elasticnet"],
            "l1_ratio": np.linspace(0.0, 1.0, 6),  # usado si penalty=elasticnet
        },
        "n_iter": 25,
    }

    # 2) Random Forest
    spaces["RandomForest"] = {
        "estimator": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 400, 800],
            "max_depth": [None, 6, 10, 16, 24],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
        "n_iter": 30,
    }

    # 3) XGBoost (opcional)
    try:
        import xgboost as xgb
        spaces["XGBoost"] = {
            "estimator": xgb.XGBClassifier(
                random_state=random_state, n_estimators=400, tree_method="hist", n_jobs=-1, eval_metric="logloss"
            ),
            "params": {
                "max_depth": [3, 4, 6, 8],
                "learning_rate": np.logspace(-3, -1, 10),
                "subsample": np.linspace(0.6, 1.0, 5),
                "colsample_bytree": np.linspace(0.6, 1.0, 5),
                "reg_lambda": np.logspace(-2, 2, 10),
            },
            "n_iter": 30,
        }
    except Exception:
        warnings.warn("XGBoost no está instalado; se omitirá.")

    # 4) LightGBM (opcional)
    try:
        import lightgbm as lgb
        spaces["LightGBM"] = {
            "estimator": lgb.LGBMClassifier(random_state=random_state, n_estimators=600, n_jobs=-1),
            "params": {
                "num_leaves": [15, 31, 63, 127],
                "max_depth": [-1, 6, 10, 16],
                "learning_rate": np.logspace(-3, -1, 10),
                "min_child_samples": [10, 20, 40, 80],
                "subsample": np.linspace(0.6, 1.0, 5),
                "colsample_bytree": np.linspace(0.6, 1.0, 5),
                "reg_lambda": np.logspace(-2, 2, 10),
            },
            "n_iter": 30,
        }
    except Exception:
        warnings.warn("LightGBM no está instalado; se omitirá.")

    return spaces

# Entrenamiento + logging MLflow
def train_with_search(model_name, estimator, param_space, X_train, y_train, X_valid, y_valid, registered_name=None):
    """RandomizedSearchCV + evaluación en VALID + logging y registro en MLflow."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_space,
        n_iter=param_space.pop("n_iter", 20) if isinstance(param_space, dict) and "n_iter" in param_space else 20,
        scoring="f1",  # métrica principal alineada al negocio
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42,
        refit=True,
    )

    # Entrenamiento
    X_train_fit = ensure_dense(X_train)
    X_valid_eval = ensure_dense(X_valid)
    search.fit(X_train_fit, y_train)

    # Logging de hiperparámetros + resultados de CV
    mlflow.log_params(search.best_params_)
    mlflow.log_metric("cv_best_score_f1", float(search.best_score_))

    # Predicciones en Valid
    best_model = search.best_estimator_
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_valid_eval)[:, 1]
    elif hasattr(best_model, "decision_function"):
        # para modelos lineales sin proba
        from sklearn.preprocessing import MinMaxScaler
        scores = best_model.decision_function(X_valid_eval).reshape(-1, 1)
        y_proba = MinMaxScaler().fit_transform(scores).ravel()
    else:
        y_proba = np.zeros_like(y_valid, dtype=float)

    y_pred = best_model.predict(X_valid_eval)

    # Métricas + artefactos
    out_dir = os.path.join("artifacts", model_name)
    metrics = eval_and_log_artifacts(y_valid, y_proba, y_pred, out_dir)

    # Log del modelo y (opcional) registro en el Model Registry
    signature = None  # podrías capturar con mlflow.models.signature infer_signature(X, y)
    try:
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=registered_name,  # si se da nombre, queda registrado
        )
    except Exception:
        # Si no hay registry accesible, al menos guardar el modelo como artefacto
        mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model", signature=signature)

    return metrics, best_model

# Main
def main(processed_dir, experiment_name, tracking_uri, register_prefix):
    # 1) MLflow config
    mlflow.set_tracking_uri(tracking_uri)  # p.ej., "http://127.0.0.1:5000"
    mlflow.set_experiment(experiment_name)

    # 2) Cargar datos procesados
    X_train, y_train, X_valid, y_valid, X_test, y_test, preproc = load_processed(processed_dir)

    # 3) Espacios de búsqueda
    spaces = get_search_spaces()

    # 4) Loop de experimentos
    for model_name, cfg in spaces.items():
        estimator = cfg["estimator"]
        params = cfg["params"]
        n_iter = cfg.get("n_iter", 20)

        print(f"\n=== Entrenando {model_name} ===")
        with mlflow.start_run(run_name=model_name):
            # Log de info base
            mlflow.log_param("algorithm", model_name)
            mlflow.log_param("n_iter_search", n_iter)

            # Entrenar y evaluar en VALID
            metrics, best_model = train_with_search(
                model_name=model_name,
                estimator=estimator,
                param_space={**params, "n_iter": n_iter},
                X_train=X_train, y_train=y_train,
                X_valid=X_valid, y_valid=y_valid,
                registered_name=f"{register_prefix}-{model_name}" if register_prefix else None
            )

            # Eval en TEST y log
            X_test_eval = ensure_dense(X_test)
            if hasattr(best_model, "predict_proba"):
                y_proba_test = best_model.predict_proba(X_test_eval)[:, 1]
            else:
                y_proba_test = None
            y_pred_test = best_model.predict(X_test_eval)

            test_metrics = {
                "test_accuracy":  float(accuracy_score(y_test, y_pred_test)),
                "test_precision": float(precision_score(y_test, y_pred_test)),
                "test_recall":    float(recall_score(y_test, y_pred_test)),
                "test_f1":        float(f1_score(y_test, y_pred_test)),
            }
            if y_proba_test is not None:
                try:
                    test_metrics["test_roc_auc"] = float(roc_auc_score(y_test, y_proba_test))
                except Exception:
                    pass
            mlflow.log_metrics(test_metrics)

            print(f"[{model_name}] VALID F1={metrics['f1']:.4f} | TEST F1={test_metrics['test_f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="fase3", help="Carpeta con splits.pkl y preprocessor.pkl")
    parser.add_argument("--experiment", type=str, default="FreshMarket-Fase4", help="Nombre del experimento en MLflow")
    parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:5000", help="URI de tracking MLflow")
    parser.add_argument("--register_prefix", type=str, default="FreshMarket", help="Prefijo de nombre de modelo a registrar")
    args = parser.parse_args()

    main(args.processed_dir, args.experiment, args.tracking_uri, args.register_prefix)
