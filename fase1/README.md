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

### Problema Principal: Predicción Inexacta de Demanda de Productos Perecederos

**Descripción del problema:**

FreshMarket enfrenta un problema crítico de gestión de inventario debido a la naturaleza perecedera de sus productos. Actualmente, el equipo de compras utiliza métodos manuales basados en experiencia e intuición para decidir qué cantidad de cada producto ordenar diariamente a los proveedores.

**Situación actual:**
- **Sobrestock:** Muchos productos se compran en exceso y se pierden por vencimiento (1-7 días de vida útil)
- **Stockouts:** Frecuentemente se agotan productos populares, perdiendo ventas y clientes
- **Sin patrones identificados:** No se han detectado tendencias por día de la semana, temporada, clima, eventos especiales

**Consecuencias:**
- Alto desperdicio de alimentos (merma del 22% mensual)
- Clientes insatisfechos por productos agotados
- Márgenes de ganancia comprimidos
- Relaciones tensas con proveedores por cambios de última hora en pedidos

**Objetivo del proyecto ML:**
Desarrollar un sistema de predicción de demanda que pronostique con 72 horas de anticipación la cantidad óptima a ordenar de cada SKU (Stock Keeping Unit), considerando:
- Histórico de ventas
- Estacionalidad y días de la semana
- Eventos y festividades
- Clima
- Promociones planificadas
- Tendencias de mercado

## 💰 Impacto Económico

### Pérdidas Actuales Cuantificadas

**A. Merma por productos vencidos:**
- Pérdida mensual: $110,000 USD
- **Pérdida anual: $1,320,000 USD (16.5% de ventas)**
- Productos más afectados: verduras de hoja verde, frutas delicadas, lácteos

**B. Ventas perdidas por stockouts:**
- Estimado de pedidos cancelados: 450 por semana
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

Con una predicción precisa (reducción de error del 60-70%):
- Reducción de merma al 8%: ahorro de $840,000 USD/año
- Reducción de stockouts en 70%: recuperación de $573,000 USD/año
- Eliminación de costos operativos extras: $168,000 USD/año
- Reducción de churn en 60%: $240,000 USD/año

**BENEFICIO ANUAL ESTIMADO: $1,821,000 USD**

**ROI esperado:** 450% en el primer año (considerando inversión de $400,000 en el proyecto)

## 👥 Stakeholders y sus Necesidades

### Stakeholder 1: CEO - María Fernanda López
**Necesidades:**
- Mejorar rentabilidad general de la empresa
- Reducir desperdicio (alineado con valores de sostenibilidad)
- Escalabilidad del negocio a nuevas regiones
- Reportes ejecutivos mensuales sobre mejoras

### Stakeholder 2: Director de Operaciones - Carlos Mendoza
**Necesidades:**
- Sistema integrado con plataforma actual de gestión de inventario
- Predicciones confiables con al menos 85% de precisión
- Alertas tempranas sobre productos críticos
- Dashboard operativo en tiempo real

### Stakeholder 3: Jefa de Compras - Ana Cristina Pérez
**Necesidades:**
- Interfaz sencilla para consultar predicciones diarias
- Recomendaciones específicas por SKU y proveedor
- Explicabilidad de las predicciones (¿por qué recomienda X cantidad?)
- Margen de error para cada predicción

### Stakeholder 4: Equipo de Compras (5 personas)
**Necesidades:**
- Reducir carga de trabajo manual
- Herramienta fácil de usar (sin conocimientos técnicos)
- Notificaciones móviles
- Acceso desde cualquier dispositivo

### Stakeholder 5: CFO - Roberto Gómez
**Necesidades:**
- ROI claro y medible del proyecto
- Reducción de costos operativos
- Métricas financieras en tiempo real
- Control de presupuesto del proyecto

### Stakeholder 6: Director de TI - Luis Hernández
**Necesidades:**
- Arquitectura escalable y mantenible
- Integración con sistemas existentes (ERP, e-commerce)
- Seguridad de datos
- Documentación técnica completa

### Stakeholder 7: Proveedores (120 proveedores)
**Necesidades:**
- Pedidos más estables y predecibles
- Menos cancelaciones de última hora
- Mejor planificación de su producción

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
- Latencia máxima de predicción: 5 segundos
- Disponibilidad del sistema: 99.5%
- Debe funcionar con datos faltantes (proveedores a veces no reportan inventario)

### B. Restricciones de Tiempo

**Timeline del proyecto:**
- **Fase 1 (Comprensión):** 2 semanas
- **Fase 2 (Preparación de datos):** 4 semanas
- **Fase 3 (Modelado):** 6 semanas
- **Fase 4 (Evaluación):** 2 semanas
- **Fase 5 (Despliegue):** 4 semanas
- **Total: 18 semanas (4.5 meses)**

**Fechas críticas:**
- Inicio del proyecto: 15 de noviembre de 2025
- Piloto funcional: 15 de febrero de 2026
- Producción completa: 1 de abril de 2026

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

Para considerar el proyecto exitoso, se deben alcanzar:

1. **Precisión del modelo:** ROC-AUC >= 0.85
2. **Reducción de merma:** De 22% a menos de 10% en 6 meses
3. **Reducción de stockouts:** De 450/semana a menos de 150/semana
4. **Adopción del sistema:** 90% del equipo de compras usando el sistema regularmente
5. **ROI:** Positivo dentro de los primeros 8 meses
6. **Tiempo de respuesta:** Predicciones generadas en menos de 5 segundos
