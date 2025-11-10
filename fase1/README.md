# Fase 1: Comprensión del Negocio (Business Understanding)

## 🏢 Empresa: FreshMarket Online

### Información Básica
- **Industria**: E-commerce de productos perecederos (alimentos frescos, frutas, verduras, lácteos, carnes)
- **Ubicación**: Guatemala, con cobertura en Ciudad de Guatemala y 5 municipios aledaños
- **Tamaño**: Empresa mediana
  - 250 empleados
  - 3 centros de distribución
  - 50,000 clientes activos
  - Ventas anuales: $8 millones USD

### Modelo de Negocio
- Plataforma de e-commerce para pedidos de productos frescos
- Entrega el mismo día o al día siguiente
- Alianzas con 120 proveedores locales (agricultores, distribuidores)
- Opera 7 días a la semana

### Contexto Actual
La empresa ha crecido 300% en los últimos 2 años, especialmente post-pandemia. Sin embargo, este crecimiento acelerado ha expuesto serias deficiencias en la gestión de inventario.

## 🎯 Problema de Negocio

### Problema Principal: Alta Tasa de Abandono de Compras por Falta de Inventario

**Descripción del problema:**

FreshMarket enfrenta un problema crítico donde los clientes abandonan sus compras frecuentemente debido a la falta de inventario de productos. Actualmente, el equipo de compras utiliza métodos manuales basados en experiencia e intuición para decidir qué cantidad de cada producto ordenar diariamente a los proveedores, lo que resulta en:

**Situación actual:**
- **Stockouts frecuentes:** Productos se agotan, causando que los clientes abandonen sus carritos
- **Sobrestock:** Productos perecederos se compran en exceso y se pierden por vencimiento (1-7 días de vida útil)
- **Sin predicción de abandono:** No se puede anticipar cuándo y por qué los clientes abandonarán sus compras
- **Sin patrones identificados:** No se han detectado tendencias por día de la semana, temporada, clima, eventos especiales

**Consecuencias:**
- Alto desperdicio de alimentos (merma del 22% mensual)
- Clientes insatisfechos por productos agotados → abandonan compras
- Pérdida de ventas y clientes (churn)
- Márgenes de ganancia comprimidos
- Relaciones tensas con proveedores por cambios de última hora en pedidos

**Objetivo del proyecto ML:**

Desarrollar un **modelo de clasificación binaria** que prediga si una compra será exitosa o abandonada, considerando:
- Nivel de inventario disponible
- Histórico de compras del cliente
- Características del pedido (productos, valor, hora)
- Productos agotados en el momento
- Factores temporales (día, hora, festividades)
- Clima y temperatura
- Promociones activas

**Variable Target:**
- `compra_exitosa = 1`: Compra completada exitosamente
- `compra_exitosa = 0`: Compra abandonada (principalmente por stockout)

**Aplicación del modelo:**
Con las predicciones del modelo, el equipo de operaciones podrá:
1. Identificar sesiones de alto riesgo de abandono en tiempo real
2. Tomar acciones preventivas (asegurar inventario, ofrecer alternativas, aplicar descuentos)
3. Optimizar el nivel de inventario basado en patrones identificados
4. Mejorar la experiencia del cliente y reducir abandonos

## 💰 Impacto Económico

### Pérdidas Actuales Cuantificadas

**A. Merma por productos vencidos:**
- Pérdida mensual: $110,000 USD
- **Pérdida anual: $1,320,000 USD (16.5% de ventas)**
- Productos más afectados: verduras de hoja verde, frutas delicadas, lácteos

**B. Ventas perdidas por stockouts (compras abandonadas):**
- Estimado de pedidos abandonados: 450 por semana
- Ticket promedio: $35 USD
- **Pérdida semanal: $15,750 USD**
- **Pérdida anual: $819,000 USD**

**C. Costos operativos adicionales:**
- Pedidos urgentes a proveedores (sobrecosto 30%): $45,000 USD/año
- Horas extras del equipo de compras: $28,000 USD/año
- Descuentos por productos próximos a vencer: $95,000 USD/año

**D. Costos de oportunidad:**
- Pérdida de clientes por mala experiencia: ~500 clientes/año
- Valor de vida del cliente (LTV): $800 USD
- **Pérdida por churn: $400,000 USD/año**

### PÉRDIDA TOTAL ANUAL: $2,707,000 USD
**Esto representa el 33.8% de las ventas anuales.**

### Beneficio Esperado con Solución ML

Con un modelo que prediga abandonos con alta precisión (F1-Score > 0.85):
- Reducción de merma al 8%: ahorro de $840,000 USD/año
- Reducción de stockouts en 70%: recuperación de $573,000 USD/año
- Eliminación de costos operativos extras: $168,000 USD/año
- Reducción de churn en 60%: $240,000 USD/año

**BENEFICIO ANUAL ESTIMADO: $1,821,000 USD**

**ROI esperado:** 450% en el primer año (considerando inversión de $400,000 en el proyecto)

### Análisis de Costos por Tipo de Error

| Tipo de Error | Predicción | Realidad | Consecuencia | Costo Unitario |
|---------------|------------|----------|--------------|----------------|
| **Falso Positivo (FP)** | Exitosa | Abandonada | No tomamos acción → Perdemos venta | **$35 USD** |
| **Falso Negativo (FN)** | Abandonada | Exitosa | Acción innecesaria → Posible sobrestock | **$5 USD** |
| **Verdadero Negativo (TN)** | Abandonada | Abandonada | Acción correcta → Salvamos venta | **+$35 USD** |
| **Verdadero Positivo (TP)** | Exitosa | Exitosa | Predicción correcta → Venta normal | **$0 USD** |

## 👥 Stakeholders y sus Necesidades

### Stakeholder 1: CEO - María Fernanda López
**Necesidades:**
- Mejorar rentabilidad general de la empresa
- Reducir desperdicio (alineado con valores de sostenibilidad)
- Escalabilidad del negocio a nuevas regiones
- Reportes ejecutivos mensuales sobre mejoras
- **KPI principal:** ROI del proyecto > 400%

### Stakeholder 2: Director de Operaciones - Carlos Mendoza
**Necesidades:**
- Sistema integrado con plataforma actual de gestión de inventario
- Predicciones confiables con F1-Score mínimo de 85%
- Alertas tempranas en tiempo real sobre sesiones de alto riesgo
- Dashboard operativo con métricas actualizadas
- **KPI principal:** Reducción de stockouts en 70%

### Stakeholder 3: Jefa de Compras - Ana Cristina Pérez
**Necesidades:**
- Interfaz sencilla para consultar predicciones
- Recomendaciones de inventario basadas en patrones del modelo
- Explicabilidad de las predicciones (¿por qué se predice abandono?)
- Alertas cuando inventario está en nivel crítico
- **KPI principal:** Reducción de merma del 22% al 10%

### Stakeholder 4: Equipo de Compras (5 personas)
**Necesidades:**
- Reducir carga de trabajo manual
- Herramienta fácil de usar (sin conocimientos técnicos)
- Notificaciones móviles de alertas críticas
- Acceso desde cualquier dispositivo
- **KPI principal:** Reducción del 50% en horas de trabajo manual

### Stakeholder 5: CFO - Roberto Gómez
**Necesidades:**
- ROI claro y medible del proyecto
- Reducción de costos operativos documentada
- Métricas financieras en tiempo real
- Control de presupuesto del proyecto
- **KPI principal:** Ahorro neto > $1.5M en primer año

### Stakeholder 6: Director de TI - Luis Hernández
**Necesidades:**
- Arquitectura escalable y mantenible
- Integración con sistemas existentes (ERP, e-commerce)
- Seguridad de datos y cumplimiento regulatorio
- Documentación técnica completa
- **KPI principal:** Uptime > 99.5%, latencia < 5 segundos

### Stakeholder 7: Proveedores (120 proveedores)
**Necesidades:**
- Pedidos más estables y predecibles
- Menos cancelaciones de última hora
- Mejor planificación de su producción
- Relación comercial más fluida
- **KPI principal:** Reducción de cambios de pedidos en 60%

## 🚧 Restricciones del Proyecto

### A. Restricciones Técnicas

**Infraestructura actual:**
- ERP: Odoo (on-premise)
- E-commerce: Shopify
- Base de datos: PostgreSQL 12
- Servidores: 2 servidores físicos en oficina central
- Sin infraestructura cloud actual

**Limitaciones:**
- Datos históricos disponibles: solo 18 meses completos
- Calidad de datos variable (inconsistencias en registros)
- Sin equipo de Data Science interno (se requiere capacitación)
- Ancho de banda limitado para procesamiento en tiempo real

**Requisitos técnicos:**
- Latencia máxima de predicción: 5 segundos (inferencia en tiempo real)
- Disponibilidad del sistema: 99.5%
- Debe funcionar con datos faltantes (proveedores a veces no reportan inventario)
- Explicabilidad del modelo (interpretable para stakeholders no técnicos)

### B. Restricciones de Tiempo

**Timeline del proyecto:**
- **Fase 1 (Comprensión del Negocio):** 2 semanas
- **Fase 2 (Comprensión de Datos):** 4 semanas
- **Fase 3 (Preparación de Datos):** 3 semanas
- **Fase 4 (Modelado):** 5 semanas
- **Fase 5 (Evaluación):** 2 semanas
- **Fase 6 (Despliegue):** 4 semanas
- **Total: 20 semanas (5 meses)**

**Fechas críticas:**
- Inicio del proyecto: 15 de noviembre de 2025
- MVP funcional: 28 de febrero de 2026
- Piloto en producción: 31 de marzo de 2026
- Producción completa: 15 de abril de 2026

**Justificación de urgencia:**
La temporada alta de ventas inicia en junio (mitad de año), por lo que el sistema debe estar estabilizado antes.

### C. Restricciones de Presupuesto

**Presupuesto total aprobado: $400,000 USD**

Distribución:
- Consultoría y desarrollo ML: $180,000
- Infraestructura cloud (Azure/AWS): $60,000 (primer año)
- Licencias de software: $25,000
- Capacitación del equipo: $45,000
- Integración con sistemas existentes: $50,000
- Contingencia (10%): $40,000

**Limitaciones presupuestarias:**
- No se puede contratar personal de ML tiempo completo
- Se debe usar mayormente tecnologías open-source
- Se priorizará MLaaS (ML as a Service) sobre desarrollo desde cero

### D. Restricciones Regulatorias y de Negocio

- **GDPR/Protección de datos:** Datos de clientes deben estar en servidores en Guatemala o USA
- **Contratos con proveedores:** Pedidos deben confirmarse con 48 horas de anticipación mínimo
- **Normas de alimentos:** Trazabilidad completa de productos (regulación local)
- **Sindicato:** Automatización no puede resultar en despidos (acuerdo laboral)
- **Ética:** El modelo no debe discriminar por zona de entrega o segmento de cliente

### E. Restricciones de Recursos Humanos

**Equipo disponible:**
- 1 Project Manager (50% dedicación)
- 1 Data Engineer (a contratar)
- 2 Desarrolladores backend (25% dedicación cada uno)
- 1 Analista de datos (75% dedicación)
- Soporte de consultoría externa según necesidad

**Limitaciones:**
- Equipo sin experiencia previa en MLOps
- Alta rotación en área de TI (30% anual)
- Necesidad de mantener operaciones actuales durante implementación

## 📊 Métricas de Éxito del Proyecto

### Métricas de Machine Learning

Para considerar el modelo técnicamente exitoso:

| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| **F1-Score** | **≥ 0.85** | **Métrica principal** - Balance entre Precision y Recall |
| **Recall** | ≥ 0.80 | Capturar la mayoría de compras exitosas reales |
| **Precision** | ≥ 0.80 | Predicciones de abandono sean acertadas |
| **ROC-AUC** | ≥ 0.85 | Capacidad de discriminación entre clases |
| **Accuracy** | ≥ 0.85 | Porcentaje de predicciones correctas |

### Métricas de Negocio

| Métrica | Baseline Actual | Objetivo | Plazo |
|---------|----------------|----------|-------|
| **Tasa de abandono** | 50% | < 20% | 6 meses |
| **Merma de productos** | 22% | < 10% | 6 meses |
| **Stockouts semanales** | 450 | < 150 | 6 meses |
| **ROI del proyecto** | - | > 400% | 12 meses |
| **Ahorro anual** | - | > $1.5M | 12 meses |
| **Adopción del sistema** | - | > 90% | 3 meses |

### Métricas de Costos (por el modelo de ML)

Basado en 10,000 predicciones mensuales:

| Métrica | Cálculo | Objetivo |
|---------|---------|----------|
| **Costo por FP** | FP × $35 | < $50,000/mes |
| **Costo por FN** | FN × $5 | < $10,000/mes |
| **Ahorro por TN** | TN × $35 | > $150,000/mes |
| **Beneficio neto** | Ahorro - Costos | > $100,000/mes |

### Métricas Operacionales

| Métrica | Objetivo |
|---------|----------|
| **Latencia de predicción** | < 5 segundos |
| **Disponibilidad del sistema** | > 99.5% |
| **Tiempo de respuesta del dashboard** | < 2 segundos |
| **Predicciones por día** | ~330 (10,000/mes) |