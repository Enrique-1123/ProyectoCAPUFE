"""
Dashboard CAPUFE - Pronóstico de Vehículos con Series de Tiempo
Proyecto Final Big Data - Enero-Junio 2025
Framework: Shiny for Python
"""

from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN Y CARGA DE DATOS
# ============================================================================

def cargar_y_preparar_datos():
    """
    Carga el CSV de CAPUFE y prepara los datos
    """
    # Intentar diferentes codificaciones
    codificaciones = ['latin-1', 'ISO-8859-1', 'cp1252', 'utf-8']
    
    for encoding in codificaciones:
        try:
            print(f"Intentando cargar con codificación: {encoding}")
            df = pd.read_csv('data/Aforos-RedPropia.csv', encoding=encoding)
            print(f"✓ Archivo cargado exitosamente con codificación: {encoding}")
            
            # Mapeo de meses a números
            meses_map = {
                'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4,
                'MAYO': 5, 'JUNIO': 6, 'JULIO': 7, 'AGOSTO': 8,
                'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
            }
            
            df['MES_NUM'] = df['MES'].map(meses_map)
            
            # Limpiar columnas numéricas
            columnas_numericas = ['AUTOS', 'MOTOS', 'AUTOBUS DE 2 EJES', 
                                  'AUTOBUS DE 3 EJES', 'AUTOBUS DE 4 EJES']
            
            for col in columnas_numericas:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Renombrar columnas para facilitar el trabajo
            df = df.rename(columns={
                'AUTOBUS DE 2 EJES': 'AUTOBUS_2_EJES',
                'AUTOBUS DE 3 EJES': 'AUTOBUS_3_EJES',
                'AUTOBUS DE 4 EJES': 'AUTOBUS_4_EJES'
            })
            
            # Crear columna de fecha
            df['FECHA'] = pd.to_datetime(
                df['AÑO'].astype(str) + '-' + 
                df['MES_NUM'].astype(str) + '-01'
            )
            
            # Ordenar por fecha
            df = df.sort_values('FECHA').reset_index(drop=True)
            
            return df
            
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print("⚠️ Archivo CSV no encontrado. Usando datos de ejemplo...")
            return generar_datos_ejemplo()
        except Exception as e:
            print(f"Error con codificación {encoding}: {e}")
            continue
    
    print("⚠️ No se pudo cargar el CSV con ninguna codificación. Usando datos de ejemplo...")
    return generar_datos_ejemplo()

def generar_datos_ejemplo():
    """
    Genera datos de ejemplo si no existe el CSV
    """
    años = [2021, 2022, 2023, 2024, 2025]
    meses = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 
             'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']
    
    data = []
    for año in años:
        for i, mes in enumerate(meses, 1):
            if año == 2025 and i > 6:  # Incluir hasta junio 2025
                break
            
            # Simular datos con tendencia y estacionalidad
            base_autos = 150000 + (año - 2021) * 8000
            estacionalidad = np.sin(i * np.pi / 6) * 10000
            
            data.append({
                'AÑO': año,
                'MES': mes,
                'MES_NUM': i,
                'AUTOS': int(base_autos + estacionalidad + np.random.randint(-5000, 5000)),
                'MOTOS': np.random.randint(8000, 15000),
                'AUTOBUS_2_EJES': np.random.randint(5000, 9000),
                'AUTOBUS_3_EJES': np.random.randint(3000, 7000),
                'AUTOBUS_4_EJES': np.random.randint(1000, 3000)
            })
    
    df = pd.DataFrame(data)
    df['FECHA'] = pd.to_datetime(df['AÑO'].astype(str) + '-' + 
                                  df['MES_NUM'].astype(str) + '-01')
    return df

# ============================================================================
# MODELO DE SERIES DE TIEMPO (ARIMA)
# ============================================================================

class ModeloSeriesTiempo:
    """Modelo de Series de Tiempo para pronóstico usando enfoque simplificado"""
    
    def __init__(self, df, tipo_vehiculo, año=None):
        self.df = df.copy()
        self.tipo_vehiculo = tipo_vehiculo
        self.año = año
        self.model = None
        self.metricas = {}
        
    def preparar_serie_temporal(self):
        """Prepara serie temporal para el modelo"""
        df_trabajo = self.df.copy()
        
        if self.año:
            df_trabajo = df_trabajo[df_trabajo['AÑO'] == self.año]
        
        # Crear serie temporal mensual
        serie = df_trabajo.groupby('FECHA')[self.tipo_vehiculo].sum().sort_index()
        return serie
    
    def entrenar_modelo_simple(self):
        """Entrena modelo simplificado de series de tiempo"""
        serie = self.preparar_serie_temporal()
        
        if len(serie) < 8:
            raise ValueError("Serie temporal demasiado corta para entrenamiento")
        
        # División 80/20
        split_idx = int(len(serie) * 0.8)
        train = serie[:split_idx]
        test = serie[split_idx:]
        
        # Método simple: promedio móvil + tendencia
        historico = train.values
        
        # Calcular tendencia
        if len(historico) > 1:
            tendencia = (historico[-1] - historico[0]) / len(historico)
        else:
            tendencia = 0
        
        # Promedio de los últimos 3 meses
        if len(historico) >= 3:
            base_pronostico = np.mean(historico[-3:])
        else:
            base_pronostico = np.mean(historico)
        
        # Pronóstico para test
        pronosticos_test = []
        for i in range(len(test)):
            pronostico = base_pronostico + (tendencia * (i + 1))
            pronosticos_test.append(max(0, pronostico))
        
        # Métricas
        mae = mean_absolute_error(test.values, pronosticos_test)
        mse = mean_squared_error(test.values, pronosticos_test)
        r2 = r2_score(test.values, pronosticos_test)
        
        self.metricas = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'train_size': len(train),
            'test_size': len(test)
        }
        
        # Guardar modelo (parámetros simples)
        self.model = {
            'base_pronostico': base_pronostico,
            'tendencia': tendencia,
            'ultimos_valores': historico[-3:] if len(historico) >= 3 else historico
        }
        
        return {
            'train': train,
            'test': test,
            'pronostico_test': pronosticos_test,
            'serie_completa': serie,
            'split_idx': split_idx
        }
    
    def pronosticar(self, meses_adelante=1):
        """Genera pronóstico"""
        if self.model is None:
            return None
        
        base = self.model['base_pronostico']
        tendencia = self.model['tendencia']
        
        pronostico = base + (tendencia * meses_adelante)
        return max(0, int(pronostico))

# ============================================================================
# RESPUESTAS A CUESTIONAMIENTOS
# ============================================================================

def responder_cuestionamientos(df):
    """Responde los 3 cuestionamientos del proyecto"""
    resultados = {}
    
    # Cuestionamiento 1: Autos para Junio 2025
    try:
        modelo_autos = ModeloSeriesTiempo(df, 'AUTOS')
        datos = modelo_autos.entrenar_modelo_simple()
        pronostico_junio = modelo_autos.pronosticar(1)
        resultados['cuestionamiento_1'] = {
            'pregunta': '¿Cuál es la cantidad de Autos esperada para el mes de Junio de 2025?',
            'respuesta': f"{pronostico_junio:,} autos",
            'detalles': 'Pronóstico generado usando modelo de series de tiempo con división 80/20'
        }
    except Exception as e:
        resultados['cuestionamiento_1'] = {
            'pregunta': '¿Cuál es la cantidad de Autos esperada para el mes de Junio de 2025?',
            'respuesta': f"Error en el pronóstico: {str(e)}",
            'detalles': 'No se pudo generar el pronóstico'
        }
    
    # Cuestionamiento 2: Vehículo más transitado 2023
    try:
        df_2023 = df[df['AÑO'] == 2023]
        tipos_vehiculos = ['AUTOS', 'MOTOS', 'AUTOBUS_2_EJES', 'AUTOBUS_3_EJES', 'AUTOBUS_4_EJES']
        
        totales = {}
        for tipo in tipos_vehiculos:
            if tipo in df_2023.columns:
                totales[tipo] = df_2023[tipo].sum()
        
        if totales:
            mas_transitado = max(totales, key=totales.get)
            nombre_vehiculo = mas_transitado.replace('_', ' ').title()
            resultados['cuestionamiento_2'] = {
                'pregunta': '¿Cuál es el tipo de vehículo más transitado durante el año 2023?',
                'respuesta': f"{nombre_vehiculo} con {totales[mas_transitado]:,} vehículos",
                'detalles': totales
            }
    except Exception as e:
        resultados['cuestionamiento_2'] = {
            'pregunta': '¿Cuál es el tipo de vehículo más transitado durante el año 2023?',
            'respuesta': f"Error en el análisis: {str(e)}",
            'detalles': {}
        }
    
    # Cuestionamiento 3: Comportamiento Autobuses 2 Ejes
    try:
        if 'AUTOBUS_2_EJES' in df.columns:
            comportamiento = df.groupby('AÑO')['AUTOBUS_2_EJES'].sum()
            tendencia = "creciente" if comportamiento.iloc[-1] > comportamiento.iloc[0] else "decreciente"
            
            resultados['cuestionamiento_3'] = {
                'pregunta': 'Comportamiento de la cantidad de Autobuses de 2 ejes durante el periodo 2021 a 2025',
                'respuesta': f"Tendencia {tendencia} con variaciones estacionales",
                'detalles': dict(comportamiento),
                'estadisticas': {
                    'total_periodo': comportamiento.sum(),
                    'promedio_anual': comportamiento.mean(),
                    'maximo': comportamiento.max(),
                    'minimo': comportamiento.min()
                }
            }
    except Exception as e:
        resultados['cuestionamiento_3'] = {
            'pregunta': 'Comportamiento de la cantidad de Autobuses de 2 ejes durante el periodo 2021 a 2025',
            'respuesta': f"Error en el análisis: {str(e)}",
            'detalles': {}
        }
    
    return resultados

# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================

app_ui = ui.page_fluid(
    # CSS personalizado
    ui.tags.style("""
        body { background-color: #f5f6fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .app-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 15px;
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .section-title {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin: 35px 0 20px 0;
            padding-bottom: 12px;
            border-bottom: 3px solid #667eea;
        }
        .forecast-result {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .forecast-value {
            font-size: 48px;
            font-weight: bold;
            margin: 15px 0;
        }
        .cuestionamiento-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
        }
        .pregunta {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 16px;
        }
        .respuesta {
            font-size: 18px;
            color: #27ae60;
            font-weight: bold;
            margin: 10px 0;
        }
    """),
    
    # Encabezado
    ui.div(
        {"class": "app-header"},
        ui.h1("🚗 Movimientos Mensuales por Tipo de Vehículo - CAPUFE", 
              style="margin: 0; font-size: 28px;"),
        ui.p("Dashboard con Series de Tiempo - Proyecto Final Big Data Enero-Junio 2025", 
             style="margin: 8px 0 0 0; opacity: 0.9; font-size: 16px;")
    ),
    
    # ========================================================================
    # SECCIÓN 0: CUESTIONAMIENTOS (NUEVA SECCIÓN)
    # ========================================================================
    ui.div(
        {"class": "section-title"},
        "❓ Respuestas a Cuestionamientos del Proyecto"
    ),
    
    ui.output_ui("cuestionamientos_display"),
    
    # ========================================================================
    # SECCIÓN 1: PRONÓSTICOS CON SERIES DE TIEMPO
    # ========================================================================
    ui.div(
        {"class": "section-title"},
        "📊 Pronósticos con Series de Tiempo"
    ),
    
    ui.layout_columns(
        # Panel de control
        ui.card(
            ui.card_header("⚙️ Configuración del Modelo", style="font-weight: bold;"),
            ui.input_select(
                "año_modelo",
                "Seleccionar año:",
                choices={"": "Todos los años", "2021": "2021", "2022": "2022", 
                        "2023": "2023", "2024": "2024", "2025": "2025"},
                selected=""
            ),
            ui.input_select(
                "tipo_vehiculo",
                "Seleccionar tipo de vehículo:",
                choices={
                    "AUTOS": "🚗 Autos",
                    "MOTOS": "🏍️ Motos",
                    "AUTOBUS_2_EJES": "🚌 Autobús de 2 ejes",
                    "AUTOBUS_3_EJES": "🚌 Autobús de 3 ejes",
                    "AUTOBUS_4_EJES": "🚌 Autobús de 4 ejes"
                },
                selected="AUTOS"
            ),
            ui.input_slider(
                "mes_pronostico",
                "Meses a pronosticar:",
                min=1,
                max=12,
                value=1,
                step=1
            ),
            ui.input_action_button(
                "btn_entrenar",
                "🚀 Generar Pronóstico",
                class_="btn-primary w-100 mt-3",
                style="padding: 12px; font-size: 16px; font-weight: bold;"
            )
        ),
        
        # Métricas del modelo
        ui.card(
            ui.card_header("📈 Métricas del Modelo", style="font-weight: bold;"),
            ui.output_ui("metricas_display")
        ),
        
        # Resultado del pronóstico
        ui.card(
            ui.card_header("🔮 Resultado del Pronóstico", style="font-weight: bold;"),
            ui.output_ui("pronostico_display")
        ),
        
        col_widths=[4, 4, 4]
    ),
    
    # Gráficos
    ui.layout_columns(
        ui.card(
            ui.card_header("📉 Serie de Tiempo y Pronóstico", style="font-weight: bold;"),
            ui.output_plot("plot_serie", height="450px")
        ),
        ui.card(
            ui.card_header("📋 Datos del Modelo", style="font-weight: bold;"),
            ui.output_data_frame("tabla_datos")
        ),
        col_widths=[7, 5]
    ),
    
    # ========================================================================
    # SECCIÓN 2: ANÁLISIS EXPLORATORIO
    # ========================================================================
    ui.div(
        {"class": "section-title"},
        "📊 Análisis Exploratorio por Año"
    ),
    
    ui.layout_columns(
        ui.card(
            ui.card_header("Filtros", style="font-weight: bold;"),
            ui.input_select(
                "año_frecuencia",
                "Seleccionar año:",
                choices=["2021", "2022", "2023", "2024", "2025"],
                selected="2023"
            ),
            ui.input_checkbox_group(
                "tipos_seleccionados",
                "Tipos de vehículos:",
                choices={
                    "AUTOS": "🚗 Autos",
                    "MOTOS": "🏍️ Motos",
                    "AUTOBUS_2_EJES": "🚌 Autobús 2 ejes",
                    "AUTOBUS_3_EJES": "🚌 Autobús 3 ejes",
                    "AUTOBUS_4_EJES": "🚌 Autobús 4 ejes"
                },
                selected=["AUTOS", "MOTOS", "AUTOBUS_2_EJES"]
            )
        ),
        ui.card(
            ui.card_header("📊 Distribución por Tipo", style="font-weight: bold;"),
            ui.output_plot("plot_frecuencias", height="400px")
        ),
        ui.card(
            ui.card_header("📊 Resumen Estadístico", style="font-weight: bold;"),
            ui.output_ui("resumen_estadistico")
        ),
        col_widths=[3, 6, 3]
    )
)

# ============================================================================
# LÓGICA DEL SERVIDOR
# ============================================================================

def server(input, output, session):
    
    # Cargar datos
    df = cargar_y_preparar_datos()
    
    # Variables reactivas
    modelo_actual = reactive.Value(None)
    datos_modelo = reactive.Value(None)
    cuestionamientos = reactive.Value(responder_cuestionamientos(df))
    
    @reactive.Effect
    @reactive.event(input.btn_entrenar)
    def entrenar_modelo():
        """Entrenar el modelo cuando se presiona el botón"""
        año = int(input.año_modelo()) if input.año_modelo() else None
        tipo = input.tipo_vehiculo()
        
        try:
            modelo = ModeloSeriesTiempo(df, tipo, año)
            datos = modelo.entrenar_modelo_simple()
            
            modelo_actual.set(modelo)
            datos_modelo.set(datos)
        except Exception as e:
            # En caso de error, mostrar mensaje
            modelo_actual.set(None)
            datos_modelo.set({'error': str(e)})
    
    @output
    @render.ui
    def cuestionamientos_display():
        """Mostrar respuestas a los cuestionamientos"""
        cues = cuestionamientos.get()
        
        html = []
        for key, cue in cues.items():
            card = ui.div(
                {"class": "cuestionamiento-card"},
                ui.div({"class": "pregunta"}, cue['pregunta']),
                ui.div({"class": "respuesta"}, cue['respuesta']),
            )
            
            # Agregar detalles si existen
            if 'detalles' in cue and cue['detalles']:
                detalles_html = "<div style='margin-top: 10px; font-size: 14px; color: #666;'><strong>Detalles:</strong><br>"
                if isinstance(cue['detalles'], dict):
                    for k, v in cue['detalles'].items():
                        detalles_html += f"{k}: {v:,}<br>" if isinstance(v, (int, float)) else f"{k}: {v}<br>"
                else:
                    detalles_html += str(cue['detalles'])
                detalles_html += "</div>"
                card.children.append(ui.HTML(detalles_html))
            
            html.append(card)
        
        return ui.div(*html)
    
    @output
    @render.ui
    def metricas_display():
        """Mostrar métricas del modelo"""
        modelo = modelo_actual.get()
        
        if modelo is None:
            return ui.div(
                ui.p("Presiona el botón para generar el pronóstico", 
                     style="text-align: center; color: #999; padding: 20px;")
            )
        
        metricas = modelo.metricas
        
        return ui.div(
            ui.div(
                {"class": "metric-card"},
                ui.div({"class": "metric-value"}, f"{metricas['r2']:.4f}"),
                ui.div({"class": "metric-label"}, "R² Score")
            ),
            ui.div(
                {"class": "metric-card"},
                ui.div({"class": "metric-value"}, f"{metricas['mae']:,.0f}"),
                ui.div({"class": "metric-label"}, "MAE")
            ),
            ui.div(
                {"class": "metric-card"},
                ui.div({"class": "metric-value"}, f"{metricas['mse']:,.0f}"),
                ui.div({"class": "metric-label"}, "MSE")
            )
        )
    
    @output
    @render.ui
    def pronostico_display():
        """Mostrar resultado del pronóstico"""
        modelo = modelo_actual.get()
        
        if modelo is None:
            datos = datos_modelo.get()
            if datos and 'error' in datos:
                return ui.div(
                    {"class": "forecast-result", "style": "background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);"},
                    ui.div({"class": "forecast-value"}, "Error"),
                    ui.p(datos['error'], style="font-size: 16px; margin: 5px 0;")
                )
            return ui.div(
                ui.p("Genera un pronóstico para ver el resultado", 
                     style="text-align: center; color: #999; padding: 20px;")
            )
        
        meses = input.mes_pronostico()
        pronostico = modelo.pronosticar(meses)
        
        meses_nombres = ['', 'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                        'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        
        return ui.div(
            {"class": "forecast-result"},
            ui.div({"class": "forecast-value"}, f"{pronostico:,.0f}"),
            ui.p(f"Pronóstico para {meses} mes(es) adelante", 
                 style="font-size: 18px; margin: 5px 0;"),
            ui.p(f"Tipo: {input.tipo_vehiculo()}", 
                 style="opacity: 0.9; margin: 5px 0;")
        )
    
    @output
    @render.plot
    def plot_serie():
        """Gráfico de serie de tiempo"""
        datos = datos_modelo.get()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if datos is None or 'error' in datos:
            ax.text(0.5, 0.5, 'Genera un pronóstico para visualizar la serie de tiempo',
                   ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')
            return fig
        
        if 'train' in datos and 'test' in datos:
            train = datos['train']
            test = datos['test']
            pronostico_test = datos['pronostico_test']
            split_idx = datos['split_idx']
            
            # Datos de entrenamiento
            ax.plot(train.index, train.values, 'o-', 
                   label='Datos Entrenamiento (80%)', 
                   linewidth=2, markersize=6, color='#2c3e50')
            
            # Datos de prueba
            ax.plot(test.index, test.values, 's-', 
                   label='Datos Prueba (20%)', 
                   linewidth=2, markersize=6, color='#e74c3c')
            
            # Predicciones en test
            ax.plot(test.index, pronostico_test, '--', 
                   label='Predicción Prueba', 
                   linewidth=2.5, color='#667eea', alpha=0.8)
            
            # Pronóstico futuro
            modelo = modelo_actual.get()
            if modelo:
                meses_adelante = input.mes_pronostico()
                pronostico_futuro = modelo.pronosticar(meses_adelante)
                
                # Última fecha + meses adelante
                ultima_fecha = test.index[-1] if len(test) > 0 else train.index[-1]
                fecha_futura = ultima_fecha + pd.DateOffset(months=meses_adelante)
                
                ax.plot(fecha_futura, pronostico_futuro, '*', 
                       label=f'Pronóstico +{meses_adelante} mes(es)', 
                       markersize=25, color='#f39c12')
        
        ax.set_xlabel('Fecha', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'Cantidad de {input.tipo_vehiculo()}', fontsize=13, fontweight='bold')
        ax.set_title(f'Serie de Tiempo: {input.tipo_vehiculo()}', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    @output
    @render.data_frame
    def tabla_datos():
        """Tabla de datos del modelo"""
        datos = datos_modelo.get()
        
        if datos is None or 'error' in datos:
            return render.DataGrid(pd.DataFrame({'Mensaje': ['Genera un pronóstico para ver los datos']}))
        
        if 'serie_completa' in datos:
            serie = datos['serie_completa']
            df_tabla = pd.DataFrame({
                'Fecha': serie.index,
                'Cantidad': serie.values
            })
            return render.DataGrid(df_tabla, width="100%", height="380px")
        
        return render.DataGrid(pd.DataFrame({'Mensaje': ['Datos no disponibles']}))
    
    @output
    @render.plot
    def plot_frecuencias():
        """Gráfico de frecuencias por tipo"""
        año = int(input.año_frecuencia())
        tipos = list(input.tipos_seleccionados())
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        if not tipos:
            ax.text(0.5, 0.5, 'Selecciona al menos un tipo de vehículo',
                   ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')
            return fig
        
        df_año = df[df['AÑO'] == año].groupby('MES_NUM').agg({
            tipo: 'sum' for tipo in tipos
        }).reset_index()
        
        x = np.arange(len(df_año))
        width = 0.7 / len(tipos)
        
        colores = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                        'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        
        for i, tipo in enumerate(tipos):
            offset = width * i - (width * len(tipos) / 2) + width / 2
            valores = df_año[tipo].values
            label = tipo.replace('_', ' ').title()
            ax.bar(x + offset, valores, width, 
                  label=label, color=colores[i % len(colores)], alpha=0.85)
        
        ax.set_xlabel('Mes', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cantidad de vehículos', fontsize=13, fontweight='bold')
        ax.set_title(f'Distribución por Tipo de Vehículo - Año {año}', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([meses_nombres[i-1] for i in df_año['MES_NUM']], rotation=0)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        return fig
    
    @output
    @render.ui
    def resumen_estadistico():
        """Resumen estadístico de los tipos seleccionados"""
        año = int(input.año_frecuencia())
        tipos = list(input.tipos_seleccionados())
        
        if not tipos:
            return ui.p("Selecciona tipos de vehículos para ver el resumen")
        
        df_año = df[df['AÑO'] == año]
        
        html = ""
        for tipo in tipos:
            if tipo in df_año.columns:
                total = df_año[tipo].sum()
                promedio = df_año[tipo].mean()
                maximo = df_año[tipo].max()
                minimo = df_año[tipo].min()
                
                html += f"""
                <div class="metric-card">
                    <h6 style="color: #667eea; margin-bottom: 10px; font-weight: bold;">
                        {tipo.replace('_', ' ').title()}
                    </h6>
                    <p style="margin: 5px 0;"><strong>Total:</strong> {total:,.0f}</p>
                    <p style="margin: 5px 0;"><strong>Promedio:</strong> {promedio:,.0f}</p>
                    <p style="margin: 5px 0;"><strong>Máximo:</strong> {maximo:,.0f}</p>
                    <p style="margin: 5px 0;"><strong>Mínimo:</strong> {minimo:,.0f}</p>
                </div>
                """
        
        return ui.HTML(html) if html else ui.p("No hay datos disponibles para los tipos seleccionados")

# Crear aplicación
app = App(app_ui, server)