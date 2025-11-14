# Informe de evaluación — FreshMarket (fase EVALUACIÓN)

**Generado:** 2025-11-14

## 1. Objetivo
Realizar una evaluación comprensiva del modelo seleccionado en la fase de evaluación del CRISP-DM: métricas, curvas ROC/PR, matriz de confusión, interpretación de las variables, validación cruzada temporal, análisis de errores y estimación de impacto económico.

## 2. Resumen ejecutivo (puntos clave)
- Conjunto de test usado: `artifacts/evaluation/test_with_preds.csv`
- Métricas principales:
  - Accuracy: **0.775**
  - Precision: **0.7065**
  - Recall (sensibilidad): **0.7083**
  - F1: **0.7074**
  - ROC AUC: **0.8445**
- Matriz de confusión (test):
  - TP = 272, TN = 503, FP = 113, FN = 112
- Estimación de impacto (costos defin.: FP=\$35, FN=\$5, TN gain=\$35):
  - Net en dataset: **\$13,090**
  - Estimación mensual (escalada): **\$130,900**
  - Estimación anual (escalada): **\$1,570,800**

**Interpretación rápida:** el modelo presenta buen poder discriminativo (AUC ~0.84) y un F1 ≈ 0.71 — rendimiento sólido para un primer despliegue en modo *shadow* o A/B antes de automatizar acciones.

## 3. Interpretabilidad — qué explican las principales features (top 10)
La importancia de features (top 10) según el modelo:

1. **num__ratio_agotados_inventario (0.274)**  
   - Significado: proporción de SKUs agotados en el inventario del comercio.  
   - Interpretación: mayor ratio → mayor probabilidad de clase positiva (por ejemplo, riesgo de churn/abandono de compra o necesidad de intervención). Es la variable más influyente.

2. **num__nivel_inventario_general (0.125)**  
   - Significado: nivel promedio de inventario.  
   - Interpretación: niveles bajos asociados a riesgo; niveles altos reducen probabilidad de evento adverso.

3. **num__productos_agotados (0.0696)**  
   - Significado: conteo absoluto de productos agotados. Complementa al ratio.

4. **num__tiempo_carga_sitio (0.0593)**  
   - Significado: latencia o tiempo de carga de la página.  
   - Interpretación: más tiempo → mayor fricción → mayor probabilidad de abandono/resultado negativo.

5. **num__temperatura (0.0507)**  
   - Significado: temperatura local (puede correlacionar con demanda o condiciones logísticas).  
   - Interpretación: dependencia no lineal posible; investigar interacción con hora/día.

6. **num__valor_promedio_item (0.0491)**  
   - Significado: precio medio por artículo.  
   - Interpretación: items más económicos o más caros pueden cambiar la propensión a conversión/abandono.

7. **num__valor_carrito (0.0406)**  
   - Significado: valor total del carrito. Usualmente correlaciona con decisiones de pago.

8. **num__ticket_promedio_historico (0.0365)**  
   - Significado: comportamiento histórico del cliente — buen predictor de comportamiento futuro.

9. **num__dia_mes (0.0357)**  
   - Significado: día del mes — patrones temporales (pago de sueldo, promociones).

10. **num__hora_pedido (0.0343)**  
    - Significado: hora del pedido — comportamiento horario (peak hours vs. off-peak).

**Recomendación de interpretación:** pasar estas importancias a SHAP (o Partial Dependence) para:
- confirmar dirección y forma (lineal / no lineal),
- detectar interacciones (por ejemplo: `tiempo_carga_sitio × valor_carrito`),
- identificar umbrales prácticos (p. ej. ratio_agotados > 0.3 → alerta).

## 4. Análisis de errores y casos edge
- Archivos: `cases_FN.csv` (falsos negativos) y `cases_FP.csv` (falsos positivos).
- Observación inicial: FP = 113 vs FN = 112 (balance razonable).  
- Acciones recomendadas:
  1. Revisar top 50 FN: ¿corresponden a segmentos de alto ticket? priorizar reducción de FN si el costo de FN es alto en ciertos segmentos.
  2. Crear reglas de negocio híbridas: para clientes con `valor_carrito` > X usar umbral más conservador.
  3. Hacer inspección manual (call center / analistas) sobre 30-50 casos representativos para entender por qué el modelo falla.

## 5. Validación temporal
- CV (TimeSeriesSplit/Stratified) (5 folds) → scores F1:  
  `[0.7072, 0.6936, 0.7035, 0.6732, 0.6782]` → variación moderada; sugiere robustez pero con drift temporal leve.
- Recomendación: programar evaluación mensual y re-entrenamiento si la media móvil de F1 cae > 0.03.

## 6. Limitaciones / riesgos
- **Compatibilidad de versiones:** el modelo fue serializado con scikit-learn 1.7.2 y en producción se evaluó con 1.5.1. Esto puede introducir diferencias sutiles. Reentrenar o alinear versiones recomendado.
- **Estimación de impacto:** la escala mensual/anual es una proyección lineal con supuestos simplificados. Debe validarse con A/B tests antes de tomar decisiones automáticas.
- **SHAP:** aún se recomienda generar SHAP global y local. Si no existe, generarlo para explicar decisiones individuales.

## 7. Próximos pasos técnicos (priorizados)
1. Optimizar umbral (threshold) para minimizar coste de negocio (buscar umbral que maximice `est_monthly_net` en validation set).  
2. Generar SHAP summary + dependence plots + force plots para top 20 casos FN/FP.  
3. Calibrar probabilidades (CalibratedClassifierCV) si se va a tomar decisiones por umbral.  
4. Implementar monitorización: F1 diario/semana, % predicciones positivas, distribución de top-3 features.  
5. Reentrenar / versionar modelo con `mlflow.sklearn.log_model(...)` y documentar la versión exacta de sklearn en `requirements.txt`.

---

**Archivos clave:**  
`artifacts/evaluation/metrics.json`, `metrics.csv`, `test_with_preds.csv`, `cases_FP.csv`, `cases_FN.csv`, `feature_importance/feature_importance.csv`, `artifacts/evaluation/evaluation_report.md`, `artifacts/evaluation/evaluation_report.md` (copias de imágenes: `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `feature_importance.png`, `shap_beeswarm.png`).
