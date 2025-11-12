"""
Fase 3 - Preparación de Datos 
incluye: limpieza, feature engineering, transformaciones y split temporal.

"""
import os
import argparse
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

RANDOM_STATE = 42



# Utilidades
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_dataset(input_path: str) -> pd.DataFrame:
    import os
    print(f"[read_dataset] intentando CSV en: {input_path}")

    # Si no se indicó ruta o no existe, busca en ubicaciones comunes
    if not input_path or not os.path.exists(input_path):
        candidates = [
            input_path,
            os.path.join("data", "raw", "freshmarket_dataset.csv"),
            os.path.join("fase2", "freshmarket_dataset.csv"),
            "freshmarket_dataset.csv",
        ]
        input_path = next((p for p in candidates if p and os.path.exists(p)), None)
        if not input_path:
            raise FileNotFoundError(
                "No se encontró el CSV. Colócalo en 'data/raw/' o 'fase2/' "
                "con el nombre 'freshmarket_dataset.csv'."
            )

    # Leer como csv
    df = pd.read_csv(input_path, parse_dates=["fecha_pedido"])
    print(f"[read_dataset] cargado CSV -> shape={df.shape}")
    return df

# reporte de estado de los datos
def quality_report(df: pd.DataFrame, title: str = "Reporte de estado inicial de los datos") -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("Shape:", df.shape)
    print("\nNulos por columna (top 12):")
    print(df.isna().sum().sort_values(ascending=False).head(12))
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nEjemplo de filas:")
    print(df.head(3))

# Validaciones iniciales
def sanity_checks_raw(df: pd.DataFrame) -> None:
    """Chequeos rápidos basados en el generador de datos."""
    required_cols = [
        # temporales
        "fecha_pedido", "dia_semana", "mes", "dia_mes", "hora_pedido",
        "es_fin_semana", "es_festivo",
        # cliente
        "cliente_id", "segmento_cliente", "compras_previas",
        "ticket_promedio_historico", "dias_desde_ultima_compra", "zona_entrega",
        # pedido
        "num_items_carrito", "incluye_perecederos", "valor_carrito", "tipo_entrega",
        # inventario
        "nivel_inventario_general", "productos_agotados", "tiempo_carga_sitio",
        # marketing
        "hay_promocion", "descuento_aplicado", "canal_adquisicion",
        # externos
        "clima", "temperatura",
        # target
        "compra_exitosa",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Rango y tipos razonables 
    assert df["nivel_inventario_general"].between(0, 100).mean() > 0.95, \
        "Muchos niveles de inventario fuera de [0,100]"
    assert df["productos_agotados"].between(0, 50).mean() > 0.95, \
        "Muchos productos_agotados fuera de [0,50]"
    assert df["tiempo_carga_sitio"].between(0.0, 10.0).mean() > 0.95, \
        "Valores raros en tiempo_carga_sitio"

    # Target binaria balanceada 
    target_ratio = df["compra_exitosa"].mean()
    if not (0.4 <= target_ratio <= 0.6):
        warnings.warn(f"Target desbalanceada (mean={target_ratio:.3f}). "
                      f"Verificar generador/periodos.")


# Limpieza
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]

    df = df.drop_duplicates()

    df["fecha_pedido"] = pd.to_datetime(df["fecha_pedido"], errors="coerce")

    for col in ["ticket_promedio_historico", "valor_carrito", "temperatura"]:
        if col in df:
            df[col] = df[col].astype(float)
            df[col] = df[col].fillna(df[col].median())

    # Rango válido en métricas claves
    df["nivel_inventario_general"] = df["nivel_inventario_general"].clip(0, 100)
    df["productos_agotados"] = df["productos_agotados"].clip(lower=0, upper=50)
    df["tiempo_carga_sitio"] = df["tiempo_carga_sitio"].clip(lower=0.0, upper=10.0)

    after = df.shape[0]
    print(f"Limpieza: filas antes={before}, después={after} (eliminados={before - after})")
    return df


# Split temporal
def temporal_split(df: pd.DataFrame,
                   train_frac: float = 0.7,
                   valid_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    train/valid/test = 70/20/10
    """
    if "fecha_pedido" not in df.columns:
        raise ValueError("Se requiere la columna 'fecha_pedido' para el split temporal.")

    df_sorted = df.sort_values("fecha_pedido").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(train_frac * n)
    valid_end = int((train_frac + valid_frac) * n)

    train = df_sorted.iloc[:train_end].copy()
    valid = df_sorted.iloc[train_end:valid_end].copy()
    test = df_sorted.iloc[valid_end:].copy()

    print(f"Split temporal -> train={train.shape}, valid={valid.shape}, test={test.shape}")
    return train, valid, test

# Feature engineering
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Señales temporales
    df["anio"] = df["fecha_pedido"].dt.year
    df["mes_num"] = df["fecha_pedido"].dt.month
    df["dia_num"] = df["fecha_pedido"].dt.day
    df["hora_num"] = df["fecha_pedido"].dt.hour
    df["fin_de_mes"] = df["fecha_pedido"].dt.is_month_end.astype(int)

    # Interacciones útiles
    df["ratio_agotados_inventario"] = df["productos_agotados"] / (df["nivel_inventario_general"] + 1.0)
    df["valor_promedio_item"] = df["valor_carrito"] / (df["num_items_carrito"] + 1.0)

    # Identificadores no predictivos
    drop_ids = ["cliente_id"]
    df = df.drop(columns=[c for c in drop_ids if c in df.columns])

    return df


# Transformaciones 
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Asegurarnos de no escalar la etiqueta ni la fecha si se colara
    for col in ["compra_exitosa", "fecha_pedido"]:
        if col in numeric_cols:
            numeric_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])
    return preprocessor


def transform_split(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    target = "compra_exitosa"

    def _split_xy(df):
        X = df.drop(columns=[target])
        y = df[target].astype(int)
        return X, y

    # Feature engineering previo a separar X/y
    train_fe = feature_engineering(train)
    valid_fe = feature_engineering(valid)
    test_fe = feature_engineering(test)

    # Separar y retirar fecha 
    drop_after_split = ["fecha_pedido"]
    train_fe = train_fe.drop(columns=[c for c in drop_after_split if c in train_fe.columns])
    valid_fe = valid_fe.drop(columns=[c for c in drop_after_split if c in valid_fe.columns])
    test_fe  = test_fe.drop(columns=[c for c in drop_after_split if c in test_fe.columns])

    X_train, y_train = _split_xy(train_fe)
    X_valid, y_valid = _split_xy(valid_fe)
    X_test,  y_test  = _split_xy(test_fe)

    preprocessor = build_preprocessor(X_train)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_valid_proc = preprocessor.transform(X_valid)
    X_test_proc  = preprocessor.transform(X_test)

    print(f"Transformación completada. Dimensiones finales:")
    print(f"  X_train: {X_train_proc.shape}, X_valid: {X_valid_proc.shape}, X_test: {X_test_proc.shape}")

    return (X_train_proc, y_train, X_valid_proc, y_valid, X_test_proc, y_test, preprocessor)


# Guardado de artefactos
def persist_artifacts(outdir: str,
                      splits_tuple,
                      preprocessor: ColumnTransformer) -> None:
    ensure_dir(outdir)

    X_train, y_train, X_valid, y_valid, X_test, y_test, _ = splits_tuple

    joblib.dump(preprocessor, os.path.join(outdir, "preprocessor.pkl"))
    joblib.dump(
        (X_train, y_train, X_valid, y_valid, X_test, y_test),
        os.path.join(outdir, "splits.pkl"),
    )
    print(f"Artefactos guardados en: {outdir}")


# Main
def main(input_path: str, outdir: str) -> None:
    print("Preparación de Datos")
    print(f"Input: {input_path}")
    print(f"Outdir: {outdir}")

    df = read_dataset(input_path)

    # Asegurar columna temporal correcta
    if "fecha_pedido" in df and not np.issubdtype(df["fecha_pedido"].dtype, np.datetime64):
        df["fecha_pedido"] = pd.to_datetime(df["fecha_pedido"], errors="coerce")

    quality_report(df, "Estado de inicial de los datos")
    sanity_checks_raw(df)

    df_clean = clean_data(df)
    quality_report(df_clean, "Estado de los datos después de la limpieza")

    train, valid, test = temporal_split(df_clean, train_frac=0.7, valid_frac=0.2)

    splits_tuple = transform_split(train, valid, test)
    _, _, _, _, _, _, preprocessor = splits_tuple

    persist_artifacts(outdir, splits_tuple, preprocessor)

    # Métrica rápida de balance en cada split
    for name, part in zip(["train", "valid", "test"], [train, valid, test]):
        ratio = part["compra_exitosa"].mean()
        print(f"Balance {name}: compra_exitosa=1 -> {ratio:.3f}")

    print("\n✓ Pipeline de preparación completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="fase2/freshmarket_dataset.csv",
                        help="Ruta al dataset (pkl o csv)")
    parser.add_argument("--outdir", type=str, default="fase3",
                        help="Directorio de salida para artefactos")
    args = parser.parse_args()

    main(args.input, args.outdir)
