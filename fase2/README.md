# Fase 2: Comprensión de los Datos (Data Understanding)

## 📁 Archivos del Proyecto
```
fase2/
├── README.md
├── crear_dataset.py
├── freshmarket_dataset.csv
└── freshmarket_dataset.pkl
```

## 📊 Dataset Sintético - FreshMarket Online

### Descripción General
- **Número de registros**: 10,000
- **Número de features**: 25 variables + 1 target
- **Período temporal**: Enero 2024 - Octubre 2025
- **Balance de clases**: Aproximadamente 50% - 50%

### Variable Target
**`compra_exitosa`** (binaria)
- **1**: Compra completada exitosamente
- **0**: Compra abandonada por stockout o falta de inventario

## 📖 Diccionario de Datos

### Variables Temporales

| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| `fecha_pedido` | datetime | Fecha y hora del pedido | 2024-01-01 a 2025-10-31 |
| `dia_semana` | categórica | Día de la semana | Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday |
| `mes` | numérica | Mes del año | 1-12 |
| `dia_mes` | numérica | Día del mes | 1-31 |
| `hora_pedido` | numérica | Hora del pedido (formato 24h) | 0-23 |
| `es_fin_semana` | binaria | Indica si es fin de semana | 0 (No), 1 (Sí) |
| `es_festivo` | binaria | Indica si es día festivo | 0 (No), 1 (Sí) |

### Variables del Cliente

| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| `cliente_id` | categórica | Identificador único del cliente | CLI_1000 a CLI_9999 |
| `segmento_cliente` | categórica | Segmento del cliente | Nuevo, Regular, VIP, Inactivo |
| `compras_previas` | numérica | Número de compras previas del cliente | 0-100 |
| `ticket_promedio_historico` | numérica | Promedio gastado en compras anteriores (USD) | 0-120 |
| `dias_desde_ultima_compra` | numérica | Días desde la última compra | 1-999 |
| `zona_entrega` | categórica | Zona de entrega del pedido | Zona 1, Zona 10, Zona 15, Mixco, Villa Nueva, San Miguel Petapa |

### Variables del Pedido

| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| `num_items_carrito` | numérica | Cantidad de items en el carrito | 1-25 |
| `incluye_perecederos` | binaria | Si incluye productos perecederos | 0 (No), 1 (Sí) |
| `valor_carrito` | numérica | Valor total del carrito (USD) | Variable |
| `tipo_entrega` | categórica | Tipo de entrega solicitada | Mismo día, Día siguiente, Programada |

### Variables de Inventario y Operaciones

| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| `nivel_inventario_general` | numérica | Nivel de inventario general del día | 0-100 |
| `productos_agotados` | numérica | Cantidad de productos sin stock | 0-30 |
| `tiempo_carga_sitio` | numérica | Tiempo de carga del sitio web (segundos) | 0.5-5.0 |

### Variables de Marketing y Promociones

| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| `hay_promocion` | binaria | Indica si hay promoción activa | 0 (No), 1 (Sí) |
| `descuento_aplicado` | numérica | Porcentaje de descuento aplicado | 0, 5, 10, 15, 20 |
| `canal_adquisicion` | categórica | Canal de adquisición del cliente | Orgánico, Redes Sociales, Email, Referido, Búsqueda Pagada |

### Variables Externas

| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| `clima` | categórica | Clima del día | Soleado, Lluvioso, Nublado |
| `temperatura` | numérica | Temperatura en Celsius | 18-35 |

### Variable Target

| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| **`compra_exitosa`** | **binaria (TARGET)** | **Indica si la compra fue completada** | **0 (Abandonada), 1 (Exitosa)** |

## 🚀 Uso

### Generar el Dataset
```bash
python crear_dataset.py
```

### Cargar el Dataset
```python
import pandas as pd

# Opción 1: Cargar desde CSV
df = pd.read_csv('freshmarket_dataset.csv', parse_dates=['fecha_pedido'])

# Opción 2: Cargar desde pickle (preserva tipos de datos)
df = pd.read_pickle('freshmarket_dataset.pkl')
```

## 📋 Resumen de Variables

- **Total de variables**: 26 (25 features + 1 target)
- **Variables numéricas**: 13
- **Variables categóricas**: 12
- **Variable target**: 1 (binaria)
