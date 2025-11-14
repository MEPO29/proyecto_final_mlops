#!/usr/bin/env python3
# fase5/model_evaluation.py
"""
Evaluación comprensiva de modelo - FreshMarket
- Extrae el mejor run del experimento MLflow (por test_f1 / cv_best_score_f1 fallback)
- Calcula métricas estándar (precision, recall, f1, roc_auc, accuracy)
- Matriz de confusión
- Curvas ROC y Precision-Recall
- Feature importance (feature_importances_ / coef_) y SHAP (si posible)
- Validación cruzada temporal si hay columna de tiempo; si no, StratifiedKFold
- Análisis de errores (lista de casos FP/FN/TN/TP con probabilidades y features)
- Evaluación de impacto de negocio (usa costos del README: FP=$35, FN=$5, TN +$35)
- Guarda resultados en artifacts/evaluation/
"""
# --- agregar al top del script (muy arriba) ---
import os
# forzar backend no interactivo para evitar errores de Tkinter en entornos headless
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")

# suprimir advertencias inconsistentes de sklearn (opcional)
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# ------------------------------------------------

import os
import argparse
import warnings
import json
import numpy as np
import pandas as pd
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
import matplotlib.pyplot as plt

# opcional SHAP
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Defaults from README (costs)
COST_FP = 35.0
COST_FN = 5.0
GAIN_TN = 35.0

# utilidades
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_fig(path):
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def choose_best_run(client, experiment_name, prefer_metric="test_f1"):
    """Busca runs en MLflow y elige la mejor por prefer_metric (mayor mejor).
    Maneja distintas versiones de MlflowClient (get_experiment_by_name / list_experiments / search_experiments).
    """
    exp = None
    # 1) intentar get_experiment_by_name (presente en muchas versiones)
    try:
        exp = client.get_experiment_by_name(experiment_name)
    except Exception:
        exp = None

    # 2) si no lo encontró, intentar list_experiments() y filtrar por nombre (si existe)
    if exp is None:
        try:
            exps = client.list_experiments()
            matches = [e for e in exps if getattr(e, "name", None) == experiment_name]
            if matches:
                exp = matches[0]
        except Exception:
            exp = None

    # 3) fallback: intentar search_experiments (algunas versiones lo exponen)
    if exp is None:
        try:
            # search_experiments devuelve lista de Experiment
            # filter_string puede variar según versión; usar una forma segura
            exps = client.search_experiments(filter_string=f"name = '{experiment_name}'")
            if exps:
                exp = exps[0]
        except Exception:
            exp = None

    if not exp:
        raise RuntimeError(f"Experimento '{experiment_name}' no encontrado en MLflow tracking.")

    # obtener runs del experimento
    runs = client.search_runs(exp.experiment_id, filter_string="", max_results=10000)
    if not runs:
        raise RuntimeError(f"No hay runs en el experimento {experiment_name} (id={exp.experiment_id}).")

    # seleccionar mejor run por métricas (busca prefer_metric, luego fallback a cv_best_score_f1 o cualquier f1)
    best = None
    best_val = -float("inf")
    for r in runs:
        metrics = r.data.metrics if hasattr(r.data, "metrics") else {}
        val = None
        # acceso directo
        if prefer_metric in metrics:
            val = metrics.get(prefer_metric)
        else:
            # buscar claves que contengan el substring prefer_metric
            for k, v in metrics.items():
                if prefer_metric in k:
                    val = v
                    break
        if val is None:
            val = metrics.get("cv_best_score_f1", None)
        if val is None:
            # buscar cualquier métrica con 'f1' en el nombre
            for k, v in metrics.items():
                if "f1" in k.lower():
                    val = v
                    break
        if val is None:
            continue
        try:
            numeric = float(val)
        except Exception:
            continue
        if numeric > best_val:
            best_val = numeric
            best = r

    # si no encontró por métricas, usa el último run (fallback)
    if best is None:
        best = runs[0]

    return best


def load_model_from_run(run):
    """Carga modelo guardado en run (intenta mlflow.pyfunc, mlflow.sklearn, o joblib)."""
    artifact_uri = run.info.artifact_uri
    # path to model artifact -> artifact_uri + "/model"
    try:
        model = mlflow.pyfunc.load_model(f"{artifact_uri}/model")
        return model, artifact_uri
    except Exception:
        # intentar cargar con joblib si existe un pkl
        local_artifact_root = artifact_uri.replace("file://", "")
        candidate = os.path.join(local_artifact_root, "model", "model.pkl")
        if os.path.exists(candidate):
            model = joblib.load(candidate)
            return model, artifact_uri
        # buscar archivos pickles en artifact uri
        for root, _, files in os.walk(local_artifact_root):
            for f in files:
                if f.endswith(".pkl") or f.endswith(".joblib"):
                    try:
                        model = joblib.load(os.path.join(root, f))
                        return model, artifact_uri
                    except Exception:
                        pass
    raise RuntimeError("No se pudo cargar el modelo desde los artifacts del run.")

def compute_metrics(y_true, y_pred, y_prob):
    out = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = None
    return out

def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    save_fig(path)

def plot_roc(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(loc="lower right")
    save_fig(path)

def plot_pr(y_true, y_prob, path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label=f"AUC-PR = {pr_auc:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend(loc="lower left")
    save_fig(path)

def feature_importance_report(model, X, out_dir):
    """Extrae feature importance si está disponible; guarda csv y gráfico."""
    ensure_dir(out_dir)
    fi = None
    if hasattr(model, "feature_importances_"):
        fi_vals = np.array(model.feature_importances_)
        fi = pd.Series(fi_vals, index=X.columns).sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        coef = np.abs(np.array(model.coef_)).ravel()
        fi = pd.Series(coef, index=X.columns).sort_values(ascending=False)
    if fi is not None:
        fi.reset_index().to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
        # gráfico top 30
        topn = fi.head(30)[::-1]
        plt.figure(figsize=(6,8))
        topn.plot(kind="barh")
        plt.title("Feature importance (top 30)")
        save_fig(os.path.join(out_dir, "feature_importance.png"))
    return fi

def shap_report(model, X, out_dir):
    """Genera análisis SHAP (si está instalado)."""
    if not HAS_SHAP:
        return None
    ensure_dir(out_dir)
    try:
        explainer = None
        if hasattr(model, "predict_proba") and "sklearn" in str(type(model)):
            # sklearn model wrapper - usar TreeExplainer o KernelExplainer
            try:
                explainer = shap.Explainer(model.predict_proba, X)  # intenta autodetect
            except Exception:
                try:
                    explainer = shap.TreeExplainer(model)
                except Exception:
                    explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:,1], shap.sample(X, 200))
        else:
            # mlflow pyfunc (callable)
            try:
                explainer = shap.KernelExplainer(lambda x: model.predict(x), shap.sample(X, 200))
            except Exception:
                pass
        if explainer is None:
            return None
        shap_values = explainer(X)
        # summary plot
        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        save_fig(os.path.join(out_dir, "shap_beeswarm.png"))
        # save shap values numeric
        try:
            shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
            shap_df.to_csv(os.path.join(out_dir, "shap_values.csv"), index=False)
        except Exception:
            pass
        return shap_values
    except Exception as e:
        warnings.warn(f"SHAP fallo: {e}")
        return None

def temporal_cv_scores(X, y, model, time_col=None, n_splits=5, scoring_fn=None):
    """Si time_col existe, ejecuta TimeSeriesSplit; de lo contrario StratifiedKFold."""
    if time_col and time_col in X.columns:
        # ordenar por tiempo
        order = np.argsort(pd.to_datetime(X[time_col]))
        Xs = X.iloc[order]; ys = y.iloc[order]
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(Xs))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))
    scores = []
    for train_idx, val_idx in splits:
        Xtr, Xv = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yv = y.iloc[train_idx], y.iloc[val_idx]
        # re-fit clone of model (assume sklearn-like)
        from sklearn.base import clone
        m = clone(model)
        try:
            m.fit(Xtr, ytr)
            if hasattr(m, "predict_proba"):
                prob = m.predict_proba(Xv)[:,1]
                ypred = (prob >= 0.5).astype(int)
            else:
                ypred = m.predict(Xv)
                prob = None
            sc = scoring_fn(yv, ypred, prob) if scoring_fn else f1_score(yv, ypred, zero_division=0)
            scores.append(sc)
        except Exception as e:
            warnings.warn(f"Error en fold temporal: {e}")
            scores.append(None)
    return scores

def scoring_business(y_true, y_pred, y_prob=None):
    """Scoring rápido: F1 fallback. Intended to be used in CV scoring_fn param."""
    return float(f1_score(y_true, y_pred, zero_division=0))

def business_impact_from_confusion(cm, dataset_size, monthly_scale=10000):
    """Calcula costo/beneficio usando costos por FP/FN/TN (README). Devuelve estimaciones mensuales y anuales."""
    # cm layout: [[tn, fp],[fn, tp]]
    tn, fp, fn, tp = cm.ravel()
    # costos en dataset
    cost_fp = fp * COST_FP
    cost_fn = fn * COST_FN
    gain_tn = tn * GAIN_TN
    net = gain_tn - (cost_fp + cost_fn)
    # escala a monthly_scale predicciones
    scale = monthly_scale / dataset_size if dataset_size > 0 else 0
    est_monthly_net = net * scale
    est_annual_net = est_monthly_net * 12
    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "net_dataset": float(net),
        "est_monthly_net": float(est_monthly_net),
        "est_annual_net": float(est_annual_net),
        "cost_fp": float(cost_fp),
        "cost_fn": float(cost_fn),
        "gain_tn": float(gain_tn)
    }

def main(args):
    ensure_dir(args.output_dir)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    print("Buscando mejor run en experimento:", args.experiment)
    best_run = choose_best_run(client, args.experiment, prefer_metric=args.prefer_metric)
    print("Best run:", best_run.info.run_id)
    # cargar modelo
    # Cargar modelo: si se pasó --model_path carga local; si no, intenta cargar desde el run seleccionado
    if args.model_path:
        if not os.path.exists(args.model_path):
            raise RuntimeError(f"model_path no encontrado: {args.model_path}")
        import joblib
        model = joblib.load(args.model_path)
        artifact_uri = None
        print("Modelo cargado desde model_path:", args.model_path)
    else:
        model, artifact_uri = load_model_from_run(best_run)
        print("Modelo cargado desde run artifacts:", artifact_uri)

    print("Modelo cargado desde:", artifact_uri)
    # cargar datos (por defecto de processed_dir como en model_training.py)
    if args.processed_dir:
        splits_path = os.path.join(args.processed_dir, "splits.pkl")
        preproc_path = os.path.join(args.processed_dir, "preprocessor.pkl")
        if os.path.exists(splits_path):
            X_train, y_train, X_valid, y_valid, X_test, y_test = joblib.load(splits_path)
            # si vienen como numpy, tratar de convertir a DataFrame con nombres placeholder
            if not hasattr(X_test, "columns"):
                # intentar recuperar feature names desde preprocessor si existe
                feature_names = None
                if os.path.exists(preproc_path):
                    try:
                        pp = joblib.load(preproc_path)
                        if hasattr(pp, "get_feature_names_out"):
                            feature_names = pp.get_feature_names_out()
                    except Exception:
                        pass
                X_test = pd.DataFrame(X_test, columns=feature_names)
                X_valid = pd.DataFrame(X_valid, columns=feature_names)
                X_train = pd.DataFrame(X_train, columns=feature_names)
            y_test = pd.Series(y_test, name="target")
            y_valid = pd.Series(y_valid, name="target")
        else:
            raise RuntimeError(f"No existe splits.pkl en {args.processed_dir}")
    else:
        raise RuntimeError("Se requiere processed_dir que contenga splits.pkl (como en model_training.py).")
    # Predicciones sobre test
    X_test_eval = X_test.copy()
    if hasattr(model, "predict_proba"):
        y_prob_test = model.predict_proba(X_test_eval)[:,1]
    else:
        try:
            y_prob_test = model.predict(X_test_eval)
        except Exception:
            y_prob_test = np.zeros(len(X_test_eval))
    y_pred_test = (y_prob_test >= args.threshold).astype(int) if y_prob_test is not None else model.predict(X_test_eval)
    # Métricas
    metrics = compute_metrics(y_test, y_pred_test, y_prob_test)
    print("Metrics:", metrics)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # Matriz confusión + plots
    cm = confusion_matrix(y_test, y_pred_test)
    plot_confusion(cm, os.path.join(args.output_dir, "confusion_matrix.png"))
    # ROC/PR
    try:
        plot_roc(y_test, y_prob_test, os.path.join(args.output_dir, "roc_curve.png"))
        plot_pr(y_test, y_prob_test, os.path.join(args.output_dir, "pr_curve.png"))
    except Exception as e:
        warnings.warn(f"Error generando curvas: {e}")
    # Feature importance
    fi = feature_importance_report(model, X_test_eval, os.path.join(args.output_dir, "feature_importance"))
    # SHAP
    if args.do_shap:
        print("Generando SHAP (puede tomar tiempo)...")
        shap_values = shap_report(model, X_test_eval, os.path.join(args.output_dir, "shap"))
    else:
        shap_values = None
    # Análisis de errores: guardar casos FP/FN/TN/TP
    df_test = X_test_eval.copy()
    df_test["y_true"] = y_test.values
    df_test["y_prob"] = y_prob_test
    df_test["y_pred"] = y_pred_test
    df_test["case_type"] = df_test.apply(lambda r: "TP" if (r.y_true==1 and r.y_pred==1) else ("TN" if (r.y_true==0 and r.y_pred==0) else ("FP" if (r.y_true==0 and r.y_pred==1) else "FN")), axis=1)
    df_test.to_csv(os.path.join(args.output_dir, "test_with_preds.csv"), index=False)
    # guardar archivos por case
    for case in ["FP","FN","TP","TN"]:
        sub = df_test[df_test["case_type"]==case]
        sub.to_csv(os.path.join(args.output_dir, f"cases_{case}.csv"), index=False)
    # Business impact
    bi = business_impact_from_confusion(cm, dataset_size=len(df_test), monthly_scale=args.monthly_predictions)
    with open(os.path.join(args.output_dir, "business_impact.json"), "w") as f:
        json.dump(bi, f, indent=2)
    print("Impacto de negocio (estimado):", bi)
    # Validación cruzada temporal (o stratified)
    print("Corriendo validación cruzada temporal / estratificada...")
    cv_scores = temporal_cv_scores(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), model, time_col=args.time_col, n_splits=args.cv_splits, scoring_fn=scoring_business)
    with open(os.path.join(args.output_dir, "cv_scores.json"), "w") as f:
        json.dump({"cv_scores": cv_scores}, f, indent=2)
    print("CV scores:", cv_scores)
    # Log artifacts + metrics a MLflow (opcional)
    try:
        run_id = best_run.info.run_id
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(args.output_dir, artifact_path="evaluation")
            mlflow.log_metrics(metrics)
            mlflow.log_param("eval_script", os.path.basename(__file__))
            print("Evaluation logged to MLflow under run", run_id)
    except Exception as e:
        warnings.warn(f"No se pudo loggear evaluación en MLflow: {e}")
import json
from pathlib import Path
import datetime
import pandas as pd

def generate_evaluation_report(output_dir: str):
    """
    Genera un reporte Markdown en output_dir/evaluation_report.md usando los artifacts generados:
    - metrics.json
    - business_impact.json
    - cv_scores.json
    - feature_importance/feature_importance.csv (si existe)
    - imágenes: roc_curve.png, pr_curve.png, confusion_matrix.png, feature_importance.png, shap_beeswarm.png (si existen)
    """
    out = Path(output_dir)
    report_path = out / "evaluation_report.md"
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    # Cargar métricas
    metrics = {}
    try:
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        metrics = {}

    # Cargar business impact
    bi = {}
    try:
        bi = json.loads((out / "business_impact.json").read_text(encoding="utf-8"))
    except Exception:
        bi = {}

    # CV scores
    cv = {}
    try:
        cv = json.loads((out / "cv_scores.json").read_text(encoding="utf-8"))
    except Exception:
        cv = {}

    # Feature importance
    fi_df = None
    fi_csv = out / "feature_importance" / "feature_importance.csv"
    if fi_csv.exists():
        try:
            fi_df = pd.read_csv(fi_csv)
        except Exception:
            fi_df = None

    # comprobar imágenes
    imgs = {
        "roc": out / "roc_curve.png",
        "pr": out / "pr_curve.png",
        "cm": out / "confusion_matrix.png",
        "fi": out / "feature_importance" / "feature_importance.png",
        "shap": out / "shap" / "shap_beeswarm.png"
    }

    # Empezar a escribir MD
    md_lines = []
    md_lines.append(f"# Evaluation report\n")
    md_lines.append(f"*Generado:* {now}\n")
    md_lines.append("## Resumen ejecutivo\n")
    if metrics:
        md_lines.append(f"- **Accuracy:** {metrics.get('accuracy', 'NA')}")
        md_lines.append(f"- **Precision:** {metrics.get('precision', 'NA')}")
        md_lines.append(f"- **Recall:** {metrics.get('recall', 'NA')}")
        md_lines.append(f"- **F1:** {metrics.get('f1', 'NA')}")
        md_lines.append(f"- **ROC AUC:** {metrics.get('roc_auc', 'NA')}\n")
    else:
        md_lines.append("- No se encontraron `metrics.json`.\n")

    md_lines.append("## Impacto de negocio (estimado)\n")
    if bi:
        md_lines.append(f"- TP: {bi.get('tp', 'NA')}, TN: {bi.get('tn', 'NA')}, FP: {bi.get('fp', 'NA')}, FN: {bi.get('fn', 'NA')}")
        md_lines.append(f"- Net en dataset: {bi.get('net_dataset', 'NA')}")
        md_lines.append(f"- Estimación mensual (escalada): {bi.get('est_monthly_net', 'NA')}")
        md_lines.append(f"- Estimación anual (escalada): {bi.get('est_annual_net', 'NA')}\n")
    else:
        md_lines.append("- No se encontró `business_impact.json`.\n")

    md_lines.append("## Curvas y Matrices\n")
    if imgs["roc"].exists():
        md_lines.append(f"### ROC Curve\n\n![]({imgs['roc'].name})\n")
    if imgs["pr"].exists():
        md_lines.append(f"### Precision-Recall Curve\n\n![]({imgs['pr'].name})\n")
    if imgs["cm"].exists():
        md_lines.append(f"### Matriz de confusión\n\n![]({imgs['cm'].name})\n")

    md_lines.append("## Importancia de features\n")
    if fi_df is not None and not fi_df.empty:
        md_lines.append("Top features (csv y gráfico están incluidos):\n")
        # tomar top 10 si hay
        topn = fi_df.head(10)
        md_lines.append(topn.to_markdown(index=False))
        md_lines.append("\n")
        if imgs["fi"].exists():
            md_lines.append(f"![Feature importance]({imgs['fi'].name})\n")
    else:
        md_lines.append("- No hay archivo de importancia de features.\n")

    md_lines.append("## SHAP (si se generó)\n")
    if imgs["shap"].exists():
        md_lines.append(f"![SHAP beeswarm]({imgs['shap'].name})\n")
    else:
        md_lines.append("- No se generó SHAP o no se encontró `shap_beeswarm.png`.\n")

    md_lines.append("## CV Scores (por fold)\n")
    if cv:
        md_lines.append("```\n")
        md_lines.append(json.dumps(cv, indent=2))
        md_lines.append("\n```\n")
    else:
        md_lines.append("- No se encontró `cv_scores.json`.\n")

    md_lines.append("## Archivos generados\n")
    all_files = sorted([p.name for p in out.iterdir() if p.is_file()])
    # también incluir subfolders útiles
    md_lines.append("- " + "\n- ".join(all_files) + "\n")

    # escribir archivo MD
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    # además copiar imágenes al mismo directorio si vienen de subfolders (para que los links funcionen)
    # (las imágenes ya están en output_dir o subfolders; si están en subfolders, copiarlas para mostrar inline)
    for key, p in imgs.items():
        if p.exists():
            dst = out / p.name
            try:
                if p.parent != out:
                    # copiar para colocarlo en output_dir root (sobrescribe)
                    import shutil
                    shutil.copy(p, dst)
            except Exception:
                pass

    print("Reporte generado en:", report_path)
    return str(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="fase3", help="Folder with splits.pkl (same structure as model_training.py)")
    parser.add_argument("--experiment", type=str, default="FreshMarket-Fase4", help="Experiment name in MLflow")
    parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:5000", help="MLflow tracking URI")
    parser.add_argument("--output_dir", type=str, default="artifacts/evaluation", help="Where to save evaluation artifacts")
    parser.add_argument("--prefer_metric", type=str, default="test_f1", help="Metric to select best run (default test_f1)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for probabilities")
    parser.add_argument("--do_shap", action="store_true", help="Run SHAP analysis (requires shap installed)")
    parser.add_argument("--time_col", type=str, default=None, help="Nombre de columna temporal si existe para CV temporal")
    parser.add_argument("--cv_splits", type=int, default=5, help="Número de folds para CV temporal/estratificada")
    parser.add_argument("--monthly_predictions", type=int, default=10000, help="Número de predicciones/mes para escalar impacto de negocio")
    parser.add_argument("--model_path", type=str, default=None, help="Ruta a un modelo local (.pkl / .joblib). Si se indica, ignora cargar desde run.")
    args = parser.parse_args()
    main(args)
    
        # generar reporte Markdown resumen
    try:
        report_path = generate_evaluation_report(args.output_dir)
        print("Evaluation report creado en:", report_path)
    except Exception as e:
        print("No se pudo generar evaluation_report.md:", e)

    