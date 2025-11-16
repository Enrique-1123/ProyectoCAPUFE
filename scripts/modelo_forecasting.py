import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ModeloSeriesTiempo:
    """
    Modelo de Series de Tiempo para pronóstico CAPUFE con ARIMA y Prophet
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.modelos_arima = {}
        self.modelos_prophet = {}
        self.metricas = {}
    
    def preparar_serie_temporal(self, tipo_vehiculo, año=None):
        """Prepara serie temporal para un tipo de vehículo"""
        df_trabajo = self.df.copy()
        
        if año:
            df_trabajo = df_trabajo[df_trabajo['AÑO'] == año]
        
        # Crear serie temporal mensual
        serie = df_trabajo.groupby('FECHA')[tipo_vehiculo].sum().sort_index()
        return serie
    
    def entrenar_arima(self, tipo_vehiculo, año=None):
        """Entrena modelo ARIMA con división 80/20"""
        serie = self.preparar_serie_temporal(tipo_vehiculo, año)
        
        if len(serie) < 12:
            raise ValueError(f"Serie muy corta para {tipo_vehiculo}. Datos insuficientes.")
        
        # División 80/20
        split_idx = int(len(serie) * 0.8)
        train = serie[:split_idx]
        test = serie[split_idx:]
        
        # Entrenar ARIMA
        try:
            modelo = ARIMA(train, order=(1, 1, 1))
            modelo_fit = modelo.fit()
            
            # Pronóstico en test
            pronostico_test = modelo_fit.forecast(steps=len(test))
            
            # Métricas
            mae = mean_absolute_error(test, pronostico_test)
            mse = mean_squared_error(test, pronostico_test)
            r2 = r2_score(test, pronostico_test)
            
            self.modelos_arima[tipo_vehiculo] = modelo_fit
            self.metricas[tipo_vehiculo] = {
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'train_size': len(train),
                'test_size': len(test)
            }
            
            return {
                'train': train,
                'test': test,
                'pronostico_test': pronostico_test,
                'serie_completa': serie,
                'modelo': 'ARIMA'
            }
            
        except Exception as e:
            print(f"Error en ARIMA para {tipo_vehiculo}: {e}")
            return None
    
    def entrenar_prophet(self, tipo_vehiculo, año=None):
        """Entrena modelo Prophet"""
        serie = self.preparar_serie_temporal(tipo_vehiculo, año)
        
        # Preparar datos para Prophet
        df_prophet = pd.DataFrame({
            'ds': serie.index,
            'y': serie.values
        })
        
        # División 80/20
        split_idx = int(len(df_prophet) * 0.8)
        train = df_prophet[:split_idx]
        test = df_prophet[split_idx:]
        
        # Entrenar Prophet
        modelo = Prophet()
        modelo.fit(train)
        
        # Pronóstico
        futuro = modelo.make_future_dataframe(periods=len(test), freq='M')
        pronostico = modelo.predict(futuro)
        
        # Filtrar solo período de test
        pronostico_test = pronostico[pronostico['ds'].isin(test['ds'])]['yhat'].values
        
        # Métricas
        mae = mean_absolute_error(test['y'], pronostico_test)
        mse = mean_squared_error(test['y'], pronostico_test)
        r2 = r2_score(test['y'], pronostico_test)
        
        self.modelos_prophet[tipo_vehiculo] = modelo
        self.metricas[tipo_vehiculo] = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'train_size': len(train),
            'test_size': len(test)
        }
        
        return {
            'train': train,
            'test': test,
            'pronostico_test': pronostico_test,
            'serie_completa': serie,
            'modelo': 'Prophet'
        }
    
    def pronosticar_proximo_mes(self, tipo_vehiculo, año=None):
        """Pronostica el próximo mes (Junio 2025)"""
        if tipo_vehiculo not in self.modelos_arima:
            resultado = self.entrenar_arima(tipo_vehiculo, año)
            if resultado is None:
                return None
        
        modelo = self.modelos_arima[tipo_vehiculo]
        pronostico = modelo.forecast(steps=1)
        
        return int(max(0, pronostico.iloc[0]))
    
    # RESPUESTAS A CUESTIONAMIENTOS
    def cuestionamiento_1_autos_junio2025(self):
        """Cuestionamiento 1: Autos esperados para Junio 2025"""
        return self.pronosticar_proximo_mes('AUTOS')
    
    def cuestionamiento_2_vehiculo_mas_transitado_2023(self):
        """Cuestionamiento 2: Vehículo más transitado en 2023"""
        df_2023 = self.df[self.df['AÑO'] == 2023]
        
        tipos_vehiculos = ['AUTOS', 'MOTOS', 'AUTOBUS_2_EJES', 
                          'AUTOBUS_3_EJES', 'AUTOBUS_4_EJES']
        
        totales = {}
        for tipo in tipos_vehiculos:
            if tipo in df_2023.columns:
                totales[tipo] = df_2023[tipo].sum()
        
        if not totales:
            return None, None, {}
        
        mas_transitado = max(totales, key=totales.get)
        return mas_transitado, totales[mas_transitado], totales
    
    def cuestionamiento_3_comportamiento_autobuses_2ejes(self):
        """Cuestionamiento 3: Comportamiento autobuses 2 ejes 2021-2025"""
        if 'AUTOBUS_2_EJES' not in self.df.columns:
            return None
        
        comportamiento = self.df.groupby(['AÑO', 'MES_NUM'])['AUTOBUS_2_EJES'].sum().reset_index()
        
        # Estadísticas resumen
        stats = {
            'total_2021_2025': comportamiento['AUTOBUS_2_EJES'].sum(),
            'promedio_mensual': comportamiento['AUTOBUS_2_EJES'].mean(),
            'maximo': comportamiento['AUTOBUS_2_EJES'].max(),
            'minimo': comportamiento['AUTOBUS_2_EJES'].min(),
            'tendencia': 'Creciente' if comportamiento['AUTOBUS_2_EJES'].iloc[-1] > comportamiento['AUTOBUS_2_EJES'].iloc[0] else 'Decreciente'
        }
        
        return comportamiento, stats

# FUNCIÓN PARA CARGAR DATOS (compatible con tu app.py)
def cargar_datos_forecasting():
    """Carga datos para el modelo de forecasting"""
    try:
        df = pd.read_csv('../data/Aforos_Clean.csv')
        df['FECHA'] = pd.to_datetime(df['FECHA'])
        return df
    except FileNotFoundError:
        print("Archivo Aforos_Clean.csv no encontrado")
        return None

# EJECUCIÓN DE PRUEBA
if __name__ == "__main__":
    print("🔍 PROBANDO MODELO DE SERIES DE TIEMPO")
    
    df = cargar_datos_forecasting()
    if df is not None:
        modelo = ModeloSeriesTiempo(df)
        
        # Probar cuestionamiento 1
        pronostico = modelo.cuestionamiento_1_autos_junio2025()
        print(f"🚗 Pronóstico Autos Junio 2025: {pronostico:,}")
        
        # Probar cuestionamiento 2
        vehiculo, total, todos = modelo.cuestionamiento_2_vehiculo_mas_transitado_2023()
        print(f"🏆 Vehículo más transitado 2023: {vehiculo} ({total:,})")
        
        # Probar cuestionamiento 3
        comp, stats = modelo.cuestionamiento_3_comportamiento_autobuses_2ejes()
        print(f"📊 Comportamiento Autobuses 2 Ejes: {stats}")