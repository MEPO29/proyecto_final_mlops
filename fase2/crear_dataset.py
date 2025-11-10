"""
Dataset Sintético para FreshMarket Online
Proyecto: Predicción de Compras Exitosas vs Abandonadas por Stockout

Contexto: Cada registro representa una sesión de compra/pedido en la plataforma.
Target: compra_exitosa (1 = completada, 0 = abandonada por falta de inventario)

Autor: Equipo MLOps FreshMarket
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configurar semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

# ============================================================================
# PARÁMETROS DE GENERACIÓN
# ============================================================================

N_RECORDS = 10000  # Número de registros a generar
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 10, 31)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def generar_fecha_aleatoria(start, end):
    """Genera una fecha aleatoria entre start y end"""
    delta = end - start
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 86400)
    return start + timedelta(days=random_days, seconds=random_seconds)

def asignar_clima_segun_mes(mes):
    """Asigna clima basado en el mes (estacionalidad Guatemala)"""
    if mes in [5, 6, 7, 8, 9, 10]:  # Temporada de lluvia
        return np.random.choice(['Lluvioso', 'Nublado', 'Soleado'], 
                                p=[0.5, 0.3, 0.2])
    else:  # Temporada seca
        return np.random.choice(['Soleado', 'Nublado', 'Lluvioso'], 
                                p=[0.6, 0.3, 0.1])

def es_dia_festivo(fecha):
    """Identifica días festivos importantes en Guatemala"""
    festivos = [
        (1, 1),   # Año nuevo
        (3, 29),  # Viernes Santo (aproximado)
        (5, 1),   # Día del trabajo
        (6, 30),  # Día del Ejército
        (9, 15),  # Independencia
        (10, 20), # Revolución
        (11, 1),  # Todos los Santos
        (12, 25), # Navidad
        (12, 31), # Fin de año
    ]
    return 1 if (fecha.month, fecha.day) in festivos else 0

def calcular_nivel_inventario_base():
    """Simula nivel de inventario general (0-100)"""
    # Inventario tiende a ser más bajo en fin de semana y días festivos
    return np.random.normal(70, 20)

# ============================================================================
# GENERACIÓN DEL DATASET
# ============================================================================

print("Generando dataset sintético para FreshMarket Online...")
print(f"Número de registros: {N_RECORDS}")

data = []

for i in range(N_RECORDS):
    # -------------------------
    # 1. DATOS TEMPORALES
    # -------------------------
    fecha_pedido = generar_fecha_aleatoria(START_DATE, END_DATE)
    dia_semana = fecha_pedido.strftime('%A')
    mes = fecha_pedido.month
    dia_mes = fecha_pedido.day
    hora_pedido = fecha_pedido.hour
    es_fin_semana = 1 if fecha_pedido.weekday() >= 5 else 0
    es_festivo = es_dia_festivo(fecha_pedido)
    
    # -------------------------
    # 2. DATOS DEL CLIENTE
    # -------------------------
    cliente_id = f"CLI_{random.randint(1000, 9999)}"
    
    # Segmento de cliente (afecta comportamiento de compra)
    segmento_cliente = np.random.choice(
        ['Nuevo', 'Regular', 'VIP', 'Inactivo'],
        p=[0.25, 0.45, 0.20, 0.10]
    )
    
    # Compras previas (histórico)
    if segmento_cliente == 'Nuevo':
        compras_previas = 0
        ticket_promedio_historico = 0
    elif segmento_cliente == 'Regular':
        compras_previas = np.random.randint(3, 20)
        ticket_promedio_historico = np.random.uniform(25, 50)
    elif segmento_cliente == 'VIP':
        compras_previas = np.random.randint(20, 100)
        ticket_promedio_historico = np.random.uniform(50, 120)
    else:  # Inactivo
        compras_previas = np.random.randint(1, 5)
        ticket_promedio_historico = np.random.uniform(15, 35)
    
    # Días desde última compra
    if compras_previas == 0:
        dias_desde_ultima_compra = 999
    else:
        dias_desde_ultima_compra = np.random.randint(1, 90)
    
    # Ubicación del cliente (zona de entrega)
    zona_entrega = np.random.choice(
        ['Zona 1', 'Zona 10', 'Zona 15', 'Mixco', 'Villa Nueva', 'San Miguel Petapa'],
        p=[0.15, 0.25, 0.20, 0.15, 0.15, 0.10]
    )
    
    # -------------------------
    # 3. DATOS DEL PEDIDO
    # -------------------------
    
    # Número de items en el carrito
    num_items_carrito = np.random.randint(1, 25)
    
    # Categorías de productos en el carrito (pueden ser múltiples)
    categorias_disponibles = [
        'Frutas', 'Verduras', 'Lácteos', 'Carnes', 
        'Panadería', 'Granos', 'Bebidas', 'Snacks'
    ]
    num_categorias = np.random.randint(1, 5)
    categorias_en_carrito = random.sample(categorias_disponibles, num_categorias)
    
    # Feature: ¿Incluye productos altamente perecederos?
    incluye_perecederos = 1 if any(cat in ['Frutas', 'Verduras', 'Lácteos', 'Carnes'] 
                                    for cat in categorias_en_carrito) else 0
    
    # Valor del carrito
    valor_carrito = num_items_carrito * np.random.uniform(2, 8)
    
    # Tipo de entrega
    tipo_entrega = np.random.choice(
        ['Mismo día', 'Día siguiente', 'Programada'],
        p=[0.5, 0.35, 0.15]
    )
    
    # -------------------------
    # 4. DATOS DE INVENTARIO Y OPERACIONES
    # -------------------------
    
    # Nivel de inventario general del día (0-100)
    nivel_inventario_general = calcular_nivel_inventario_base()
    
    # Ajustes por día de la semana (inventario más bajo domingo-lunes)
    if dia_semana in ['Sunday', 'Monday']:
        nivel_inventario_general -= 15
    
    # Ajuste por hora del día (inventario baja durante el día)
    nivel_inventario_general -= (hora_pedido / 24) * 10
    
    # Asegurar que esté en rango válido
    nivel_inventario_general = max(0, min(100, nivel_inventario_general))
    
    # Número de productos agotados en el momento del pedido
    if nivel_inventario_general > 70:
        productos_agotados = np.random.randint(0, 5)
    elif nivel_inventario_general > 40:
        productos_agotados = np.random.randint(3, 15)
    else:
        productos_agotados = np.random.randint(10, 30)
    
    # Tiempo de carga del sitio (segundos) - puede afectar abandono
    tiempo_carga_sitio = np.random.uniform(0.5, 5.0)
    
    # -------------------------
    # 5. DATOS DE MARKETING Y PROMOCIONES
    # -------------------------
    
    # ¿Hay promoción activa?
    hay_promocion = np.random.choice([0, 1], p=[0.7, 0.3])
    
    # Descuento aplicado (%)
    if hay_promocion:
        descuento_aplicado = np.random.choice([5, 10, 15, 20], p=[0.4, 0.3, 0.2, 0.1])
    else:
        descuento_aplicado = 0
    
    # Canal de adquisición
    canal_adquisicion = np.random.choice(
        ['Orgánico', 'Redes Sociales', 'Email', 'Referido', 'Búsqueda Pagada'],
        p=[0.35, 0.25, 0.15, 0.15, 0.10]
    )
    
    # -------------------------
    # 6. DATOS EXTERNOS
    # -------------------------
    
    # Clima del día
    clima = asignar_clima_segun_mes(mes)
    
    # Temperatura (Celsius) - Guatemala tiene clima templado
    if mes in [3, 4, 5]:  # Meses más calientes
        temperatura = np.random.uniform(25, 35)
    else:
        temperatura = np.random.uniform(18, 28)
    
    # -------------------------
    # 7. VARIABLE TARGET: COMPRA EXITOSA
    # -------------------------
    
    # Lógica para determinar si la compra fue exitosa o abandonada
    # Basado en múltiples factores con pesos realistas
    
    probabilidad_exito = 0.5  # Base
    
    # Factor 1: Inventario (más importante)
    if nivel_inventario_general > 70:
        probabilidad_exito += 0.25
    elif nivel_inventario_general < 40:
        probabilidad_exito -= 0.30
    
    # Factor 2: Productos agotados
    if productos_agotados == 0:
        probabilidad_exito += 0.15
    elif productos_agotados > 10:
        probabilidad_exito -= 0.25
    
    # Factor 3: Segmento de cliente
    if segmento_cliente == 'VIP':
        probabilidad_exito += 0.10
    elif segmento_cliente == 'Nuevo':
        probabilidad_exito -= 0.05
    
    # Factor 4: Perecederos (más sensibles a stockout)
    if incluye_perecederos and productos_agotados > 5:
        probabilidad_exito -= 0.15
    
    # Factor 5: Día de la semana
    if es_fin_semana:
        probabilidad_exito -= 0.05  # Más demanda = más stockouts
    
    # Factor 6: Promoción
    if hay_promocion:
        probabilidad_exito += 0.08
    
    # Factor 7: Tiempo de carga
    if tiempo_carga_sitio > 3:
        probabilidad_exito -= 0.10
    
    # Factor 8: Valor del carrito (carritos grandes más susceptibles a faltantes)
    if valor_carrito > 80:
        probabilidad_exito -= 0.08
    
    # Asegurar que esté en rango [0, 1]
    probabilidad_exito = max(0.05, min(0.95, probabilidad_exito))
    
    # Generar target binario
    compra_exitosa = 1 if random.random() < probabilidad_exito else 0
    
    # -------------------------
    # AGREGAR REGISTRO AL DATASET
    # -------------------------
    
    registro = {
        # Temporales
        'fecha_pedido': fecha_pedido,
        'dia_semana': dia_semana,
        'mes': mes,
        'dia_mes': dia_mes,
        'hora_pedido': hora_pedido,
        'es_fin_semana': es_fin_semana,
        'es_festivo': es_festivo,
        
        # Cliente
        'cliente_id': cliente_id,
        'segmento_cliente': segmento_cliente,
        'compras_previas': compras_previas,
        'ticket_promedio_historico': round(ticket_promedio_historico, 2),
        'dias_desde_ultima_compra': dias_desde_ultima_compra,
        'zona_entrega': zona_entrega,
        
        # Pedido
        'num_items_carrito': num_items_carrito,
        'incluye_perecederos': incluye_perecederos,
        'valor_carrito': round(valor_carrito, 2),
        'tipo_entrega': tipo_entrega,
        
        # Inventario
        'nivel_inventario_general': round(nivel_inventario_general, 2),
        'productos_agotados': productos_agotados,
        'tiempo_carga_sitio': round(tiempo_carga_sitio, 2),
        
        # Marketing
        'hay_promocion': hay_promocion,
        'descuento_aplicado': descuento_aplicado,
        'canal_adquisicion': canal_adquisicion,
        
        # Externos
        'clima': clima,
        'temperatura': round(temperatura, 1),
        
        # TARGET
        'compra_exitosa': compra_exitosa
    }
    
    data.append(registro)
    
    # Progress bar simple
    if (i + 1) % 1000 == 0:
        print(f"Generados {i + 1}/{N_RECORDS} registros...")

# ============================================================================
# CREAR DATAFRAME
# ============================================================================

df = pd.DataFrame(data)

print("\n✓ Dataset generado exitosamente!")
print(f"Dimensiones: {df.shape}")
print(f"\nBalance de la variable target:")
print(df['compra_exitosa'].value_counts(normalize=True))

# ============================================================================
# GUARDAR DATASET
# ============================================================================

# Guardar en CSV
df.to_csv('freshmarket_dataset.csv', index=False)
print("\n✓ Dataset guardado como 'freshmarket_dataset.csv'")

# Guardar también en formato pickle para preservar tipos de datos
df.to_pickle('freshmarket_dataset.pkl')
print("✓ Dataset guardado como 'freshmarket_dataset.pkl'")

print("\n" + "="*70)
print("RESUMEN DEL DATASET GENERADO")
print("="*70)
print(df.info())
print("\n" + "="*70)
print("PRIMERAS 5 FILAS")
print("="*70)
print(df.head())
print("\n" + "="*70)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*70)
print(df.describe())