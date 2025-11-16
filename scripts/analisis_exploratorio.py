import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Intentar cargar los datos con diferentes codificaciones
def cargar_datos():
    codificaciones = ['latin-1', 'ISO-8859-1', 'cp1252', 'utf-8']
    
    for codificacion in codificaciones:
        try:
            print(f"Intentando con codificación: {codificacion}")
            df = pd.read_csv('../data/Aforos-RedPropia.csv', encoding=codificacion)
            print(f"✓ Archivo cargado exitosamente con codificación: {codificacion}")
            return df
        except UnicodeDecodeError as e:
            print(f"✗ Error con {codificacion}: {e}")
        except Exception as e:
            print(f"✗ Error inesperado con {codificacion}: {e}")
    
    # Si ninguna codificación funciona, intentar con engine alternativo
    try:
        print("Intentando con engine='python'...")
        df = pd.read_csv('../data/Aforos-RedPropia.csv', encoding='latin-1', engine='python')
        print("✓ Archivo cargado con engine='python'")
        return df
    except Exception as e:
        print(f"✗ Error crítico: {e}")
        return None

# Cargar los datos
print("=" * 60)
print("CARGANDO ARCHIVO CSV")
print("=" * 60)
df = cargar_datos()

if df is None:
    print("No se pudo cargar el archivo. Saliendo...")
    exit()

# Mostrar información básica
print("\n" + "=" * 60)
print("INFORMACIÓN DEL DATASET")
print("=" * 60)
print(f"Dimensiones: {df.shape}")
print(f"\nPrimeras filas:")
print(df.head())
print(f"\nColumnas: {df.columns.tolist()}")
print(f"\nTipos de datos:")
print(df.dtypes)
print(f"\nValores nulos:")
print(df.isnull().sum())

# Limpiar y preparar los datos
def preparar_datos(df):
    """
    Función para limpiar y preparar el dataset
    """
    # Crear una copia
    df_clean = df.copy()
    
    # Mostrar valores únicos en la columna MES para debug
    print(f"\nValores únicos en columna MES: {df_clean['MES'].unique()}")
    
    # Convertir MES a número (mapeo de meses)
    meses_map = {
        'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4,
        'MAYO': 5, 'JUNIO': 6, 'JULIO': 7, 'AGOSTO': 8,
        'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
    }
    
    df_clean['MES_NUM'] = df_clean['MES'].map(meses_map)
    
    # Verificar si hay meses no mapeados
    meses_no_mapeados = df_clean[df_clean['MES_NUM'].isnull()]['MES'].unique()
    if len(meses_no_mapeados) > 0:
        print(f"Advertencia: Meses no mapeados: {meses_no_mapeados}")
    
    # Convertir columnas numéricas (eliminar comas)
    columnas_numericas = ['AUTOS', 'MOTOS', 'AUTOBUS DE 2 EJES', 
                          'AUTOBUS DE 3 EJES', 'AUTOBUS DE 4 EJES',
                          'CAMIONES DE 2 EJES', 'CAMIONES DE 3 EJES',
                          'CAMIONES DE 4 EJES', 'CAMIONES DE 5 EJES',
                          'CAMIONES DE 6 EJES', 'TOTAL']
    
    # Filtrar solo las columnas que existen en el dataframe
    columnas_numericas = [col for col in columnas_numericas if col in df_clean.columns]
    print(f"Columnas numéricas a procesar: {columnas_numericas}")
    
    for col in columnas_numericas:
        if col in df_clean.columns:
            # Convertir a string, eliminar comas y caracteres especiales
            df_clean[col] = df_clean[col].astype(str).str.replace(',', '').str.replace('"', '').str.replace("'", "")
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Crear columna de fecha
    df_clean['FECHA'] = pd.to_datetime(
        df_clean['AÑO'].astype(str) + '-' + 
        df_clean['MES_NUM'].astype(str) + '-01',
        errors='coerce'
    )
    
    # Verificar fechas creadas
    fechas_invalidas = df_clean['FECHA'].isnull().sum()
    if fechas_invalidas > 0:
        print(f"Advertencia: {fechas_invalidas} fechas no pudieron ser creadas")
    
    # Ordenar por fecha
    df_clean = df_clean.sort_values('FECHA').reset_index(drop=True)
    
    return df_clean

# Aplicar limpieza
df_clean = preparar_datos(df)

print("\n" + "=" * 60)
print("DATOS DESPUÉS DE LIMPIEZA")
print("=" * 60)
print(f"\nDimensiones: {df_clean.shape}")
if 'AUTOS' in df_clean.columns:
    print(f"\nEstadísticas de AUTOS:")
    print(df_clean['AUTOS'].describe())

# Análisis por año y tipo de vehículo
print("\n" + "=" * 60)
print("ANÁLISIS POR AÑO")
print("=" * 60)

# Columnas para el resumen (solo las que existen)
columnas_resumen = ['AUTOS', 'MOTOS', 'AUTOBUS DE 2 EJES', 
                   'AUTOBUS DE 3 EJES', 'AUTOBUS DE 4 EJES', 'TOTAL']
columnas_resumen = [col for col in columnas_resumen if col in df_clean.columns]

resumen_anual = df_clean.groupby('AÑO')[columnas_resumen].sum().round(0)
print(resumen_anual)

# Identificar años únicos disponibles
print(f"\n\nAños disponibles: {sorted(df_clean['AÑO'].unique())}")
print(f"Meses disponibles: {sorted(df_clean['MES_NUM'].dropna().unique())}")

# Análisis de tendencias de AUTOS
if 'AUTOS' in df_clean.columns:
    print("\n" + "=" * 60)
    print("TENDENCIA DE AUTOS POR AÑO")
    print("=" * 60)

    tendencia_autos = df_clean.groupby('AÑO')['AUTOS'].agg(['sum', 'mean', 'std']).round(2)
    print(tendencia_autos)

# Guardar datos limpios
try:
    df_clean.to_csv('../data/Aforos_Clean.csv', index=False, encoding='utf-8')
    print("\n✓ Datos limpios guardados en '../data/Aforos_Clean.csv'")
except Exception as e:
    print(f"✗ Error al guardar: {e}")

# Visualización básica
if 'AUTOS' in df_clean.columns and 'AÑO' in df_clean.columns:
    plt.figure(figsize=(14, 6))

    # Gráfico 1: Total de autos por año
    plt.subplot(1, 2, 1)
    autos_por_ano = df_clean.groupby('AÑO')['AUTOS'].sum()
    plt.bar(autos_por_ano.index, autos_por_ano.values, color='steelblue', alpha=0.7)
    plt.title('Total de Autos por Año', fontsize=14, fontweight='bold')
    plt.xlabel('Año')
    plt.ylabel('Total de Autos')
    plt.grid(axis='y', alpha=0.3)

    # Gráfico 2: Serie de tiempo de autos
    plt.subplot(1, 2, 2)
    if 'FECHA' in df_clean.columns:
        autos_mensual = df_clean.groupby('FECHA')['AUTOS'].sum()
        plt.plot(autos_mensual.index, autos_mensual.values, marker='o', 
                 linewidth=2, markersize=4, color='darkgreen')
        plt.title('Serie de Tiempo: Autos Mensuales', fontsize=14, fontweight='bold')
        plt.xlabel('Fecha')
        plt.ylabel('Total de Autos')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    try:
        plt.savefig('../data/analisis_exploratorio.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico guardado en '../data/analisis_exploratorio.png'")
    except Exception as e:
        print(f"✗ Error al guardar gráfico: {e}")

    plt.show()

print("\n" + "=" * 60)
print("✓ ANÁLISIS EXPLORATORIO COMPLETADO")
print("=" * 60)