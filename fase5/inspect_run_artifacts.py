# fase5/inspect_run_artifacts.py
import os
import sys
import joblib
import traceback

def list_and_try_load(root_dirs, run_id):
    found = []
    for root in root_dirs:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if run_id in dirpath:
                found.append(dirpath)
                print("=== Encontrado run folder:", dirpath)
                # listar contenido inmediato
                for f in filenames:
                    print("   -", f)
                # intentar cargar archivos candidatas
                for f in filenames:
                    fpath = os.path.join(dirpath, f)
                    if f.lower().endswith((".pkl", ".joblib")):
                        print(f"\nIntentando cargar {fpath} con joblib...")
                        try:
                            m = joblib.load(fpath)
                            print(" -> Cargado OK con joblib. Tipo:", type(m))
                        except Exception as e:
                            print(" -> FALLÓ joblib:", e)
                            traceback.print_exc()
                # también revisar subcarpetas comunes: model, models, artifacts, outputs
                for cand_dir in ("model", "models", "artifacts", "outputs"):
                    cand = os.path.join(dirpath, cand_dir)
                    if os.path.exists(cand):
                        print(f"  -> Contiene carpeta '{cand_dir}':")
                        for r, ds, fs in os.walk(cand):
                            for x in fs:
                                print("     *", os.path.join(r, x))
    return found

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python fase5/inspect_run_artifacts.py <run_id>")
        sys.exit(1)
    run_id = sys.argv[1].strip()
    roots = ["mlruns", "mlartifacts", "models"]
    print("Buscando run_id:", run_id, "en:", roots)
    found = list_and_try_load(roots, run_id)
    if not found:
        print("\nNo se encontró ninguna carpeta que contenga el run_id. Revisa si el run_id es correcto o si los artifacts están en otra ubicación.")
    else:
        print("\nInspección completada. Si ves un archivo .pkl / carpeta 'model' que cargó OK, cópialo o apunta model_evaluation.py a esa ruta.")
