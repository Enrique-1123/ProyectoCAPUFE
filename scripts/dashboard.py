from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import io

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def cargar_datos():
    """Cargar y preparar los datos"""
    # Simulación de datos - en producción cargarías desde el CSV
    # Para shinylive, necesitamos datos embebidos
    
    # Crear datos de ejemplo basados en el CSV
    data = {
        'AÑO': [],
        'MES': [],
        'MES_NUM': [],
        'AUTOS': [],
        'MOTOS': [],
        'AUTOBUS_2_EJES': [],
        'AUTOBUS_3_EJES': [],
        'AUTOBUS_4_EJES': []
    }
    
    # Datos de ejemplo (reemplazar con datos reales del CSV)
    años = [2021, 2022, 2023, 2024, 2025]
    meses = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 
             'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']
    
    for año in años:
        for i, mes in enumerate(meses, 1):
            if año == 2025 and i > 3:  # Solo hasta marzo 2025
                break
            
            data['AÑO'].append(año)
            data['MES'].append(mes)
            data['MES_NUM'].append(i)
            # Valores simulados con tendencia
            base = 150000 + (año - 2021) * 5000
            data['AUTOS'].append(base + np.random.randint(-10000, 10000))
            data['MOTOS'].append(np.random.randint(5000, 15000))
            data['AUTOBUS_2_EJES'].append(np.random.randint(3000, 8000))
            data['AUTOBUS_3_EJES'].append(np.random.randint(2000, 6000))
            data['AUTOBUS_4_EJES'].append(np.random.randint(1000, 4000))
    
    df = pd.DataFrame(data)
    return df

def entrenar_modelo_forecast(df, tipo_vehiculo, año_filtro=None):
    """Entrenar modelo de forecasting"""
    df_trabajo = df.copy()
    
    if año_filtro:
        df_trabajo = df_trabajo[df_trabajo['AÑO'] == año_filtro]
    
    # Preparar datos
    df_trabajo['PERIODO'] = range(len(df_trabajo))
    X = df_trabajo[['PERIODO', 'MES_NUM']].values
    y = df_trabajo[tipo_vehiculo].values
    
    # Entrenar modelo
    model = LinearRegression()
    model.fit(X, y)
    
    return model, X, y, df_trabajo

def pronosticar(model, ultimo_periodo, mes_futuro):
    """Generar pronóstico"""
    X_futuro = np.array([[ultimo_periodo + 1, mes_futuro]])
    return max(0, model.predict(X_futuro)[0])

# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================

app_ui = ui.page_fluid(
    # Estilos CSS personalizados
    ui.tags.style("""
        .app-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 20px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .section-title {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        .control-panel {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    """),
    
    # Encabezado
    ui.div(
        {"class": "app-header"},
        ui.h1("Movimientos mensuales por tipo de vehículo en la red CAPUFE", 
              style="margin: 0; font-size: 28px;"),
        ui.p("Dashboard (framework Shiny) Proyecto: Enero-Junio /2025", 
             style="margin: 5px 0 0 0; opacity: 0.9;")
    ),
    
    # ========================================================================
    # SECCIÓN 1: PRONÓSTICOS Y ANÁLISIS
    # ========================================================================
    ui.div(
        {"class": "section-title"},
        "📊 Análisis y Pronósticos con Series de Tiempo"
    ),
    
    ui.div(
        {"class": "control-panel"},
        ui.layout_columns(
            ui.card(
                ui.card_header("Configuración del Pronóstico"),
                ui.input_select(
                    "año_pronostico",
                    "Seleccionar año:",
                    choices=["2021", "2022", "2023", "2024", "2025"],
                    selected="2023"
                ),
                ui.input_select(
                    "tipo_vehiculo",
                    "Seleccionar tipo:",
                    choices={
                        "AUTOS": "Autos",
                        "MOTOS": "Motos",
                        "AUTOBUS_2_EJES": "Autobús de 2 ejes",
                        "AUTOBUS_3_EJES": "Autobús de 3 ejes",
                        "AUTOBUS_4_EJES": "Autobús de 4 ejes"
                    },
                    selected="AUTOS"
                ),
                ui.input_slider(
                    "mes_pronostico",
                    "Período de tiempo (Meses):",
                    min=1,
                    max=12,
                    value=1,
                    step=1
                ),
                ui.input_action_button(
                    "btn_pronosticar",
                    "Generar Pronóstico",
                    class_="btn-primary w-100"
                )
            ),
            
            ui.card(
                ui.card_header("Estadísticas del Modelo"),
                ui.output_ui("metricas_modelo")
            ),
            
            ui.card(
                ui.card_header("Resultado del Pronóstico"),
                ui.output_ui("resultado_pronostico")
            ),
            
            col_widths=[4, 4, 4]
        )
    ),
    
    ui.layout_columns(
        ui.card(
            ui.card_header("Serie de Tiempo y Pronóstico"),
            ui.output_plot("plot_serie_tiempo", height="400px")
        ),
        ui.card(
            ui.card_header("Datos del Conjunto de Entrenamiento"),
            ui.output_data_frame("tabla_datos")
        ),
        col_widths=[7, 5]
    ),
    
    # ========================================================================
    # SECCIÓN 2: FRECUENCIAS POR TIPO DE VEHÍCULO
    # ========================================================================
    ui.div(
        {"class": "section-title"},
        "📈 Frecuencias de tipos de vehículo por año"
    ),
    
    ui.div(
        {"class": "control-panel"},
        ui.layout_columns(
            ui.input_select(
                "año_visualizacion",
                "Seleccionar año:",
                choices=["2021", "2022", "2023", "2024", "2025"],
                selected="2021"
            ),
            ui.input_checkbox_group(
                "tipos_vehiculos",
                "Tipos de vehículos:",
                choices={
                    "AUTOS": "Autos",
                    "MOTOS": "Motos",
                    "AUTOBUS_2_EJES": "Autobús de 2 ejes",
                    "AUTOBUS_3_EJES": "Autobús de 3 ejes",
                    "AUTOBUS_4_EJES": "Autobús de 4 ejes"
                },
                selected=["AUTOS"]
            ),
            col_widths=[3, 9]
        )
    ),
    
    ui.layout_columns(
        ui.card(
            ui.card_header("Gráfico de Barras por Tipo de Vehículo"),
            ui.output_plot("plot_frecuencias", height="400px")
        ),
        ui.card(
            ui.card_header("Resumen Estadístico"),
            ui.output_ui("resumen_tipos")
        ),
        col_widths=[8, 4]
    )
)

# ============================================================================
# LÓGICA DEL SERVIDOR
# ============================================================================

def server(input, output, session):
    
    # Cargar datos
    df = cargar_datos()
    
    # Estado reactivo para el modelo
    modelo_actual = reactive.Value(None)
    datos_modelo = reactive.Value(None)
    
    @reactive.Effect
    @reactive.event(input.btn_pronosticar)
    def actualizar_modelo():
        """Entrenar modelo cuando se presiona el botón"""
        año = int(input.año_pronostico())
        tipo = input.tipo_vehiculo()
        
        model, X, y, df_trabajo = entrenar_modelo_forecast(df, tipo, año)
        
        modelo_actual.set(model)
        datos_modelo.set({
            'X': X,
            'y': y,
            'df': df_trabajo,
            'tipo': tipo,
            'año': año
        })
    
    @output
    @render.ui
    def metricas_modelo():
        """Mostrar métricas del modelo"""
        datos = datos_modelo.get()
        if datos is None:
            return ui.div(
                {"class": "metric-card"},
                ui.p("Presiona 'Generar Pronóstico' para ver las métricas")
            )
        
        model = modelo_actual.get()
        X, y = datos['X'], datos['y']
        
        # Calcular métricas
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        r2 = model.score(X, y)
        
        return ui.div(
            ui.div({"class": "metric-card"},
                   ui.div({"class": "metric-value"}, f"{r2:.4f}"),
                   ui.div({"class": "metric-label"}, "R² Score")),
            ui.div({"class": "metric-card"},
                   ui.div({"class": "metric-value"}, f"{mae:,.0f}"),
                   ui.div({"class": "metric-label"}, "MAE")),
            ui.div({"class": "metric-card"},
                   ui.div({"class": "metric-value"}, f"{mse:,.0f}"),
                   ui.div({"class": "metric-label"}, "MSE"))
        )
    
    @output
    @render.ui
    def resultado_pronostico():
        """Mostrar resultado del pronóstico"""
        model = modelo_actual.get()
        datos = datos_modelo.get()
        
        if model is None or datos is None:
            return ui.div(
                {"class": "metric-card"},
                ui.p("Genera un pronóstico para ver los resultados")
            )
        
        ultimo_periodo = datos['X'][-1, 0]
        mes_futuro = input.mes_pronostico()
        
        pronostico = pronosticar(model, ultimo_periodo, mes_futuro)
        tipo_label = input.tipo_vehiculo()
        
        return ui.div(
            ui.div(
                {"class": "metric-card", 
                 "style": "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;"},
                ui.div({"class": "metric-value", "style": "color: white;"}, 
                       f"{pronostico:,.0f}"),
                ui.div({"class": "metric-label", "style": "color: rgba(255,255,255,0.9);"}, 
                       f"Pronóstico: {tipo_label}"),
                ui.p(f"Mes: {input.mes_pronostico()}", 
                     style="margin-top: 10px; opacity: 0.9;")
            )
        )
    
    @output
    @render.plot
    def plot_serie_tiempo():
        """Gráfico de serie de tiempo"""
        import matplotlib.pyplot as plt
        
        datos = datos_modelo.get()
        if datos is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'Genera un pronóstico para ver la gráfica', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        model = modelo_actual.get()
        X, y = datos['X'], datos['y']
        tipo = datos['tipo']
        
        # Predicciones
        y_pred = model.predict(X)
        
        # Pronóstico futuro
        ultimo_periodo = X[-1, 0]
        mes_futuro = input.mes_pronostico()
        X_futuro = np.array([[ultimo_periodo + 1, mes_futuro]])
        y_futuro = model.predict(X_futuro)
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(X[:, 0], y, 'o-', label='Datos Reales', 
               linewidth=2, markersize=6, color='#2c3e50')
        ax.plot(X[:, 0], y_pred, '--', label='Predicción del Modelo', 
               linewidth=2, color='#667eea')
        ax.plot(X_futuro[:, 0], y_futuro, '*', label='Pronóstico', 
               markersize=20, color='#e74c3c')
        
        ax.set_xlabel('Período', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Cantidad', fontsize=12, fontweight='bold')
        ax.set_title(f'Serie de Tiempo: {tipo}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @output
    @render.data_frame
    def tabla_datos():
        """Tabla de datos"""
        datos = datos_modelo.get()
        if datos is None:
            return pd.DataFrame({'Mensaje': ['Genera un pronóstico para ver los datos']})
        
        df_tabla = datos['df'][['AÑO', 'MES', datos['tipo']]].head(20)
        return render.DataGrid(df_tabla, width="100%", height="350px")
    
    @output
    @render.plot
    def plot_frecuencias():
        """Gráfico de frecuencias por tipo"""
        import matplotlib.pyplot as plt
        
        año = int(input.año_visualizacion())
        tipos = list(input.tipos_vehiculos())
        
        if not tipos:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'Selecciona al menos un tipo de vehículo', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        df_año = df[df['AÑO'] == año]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x = np.arange(len(df_año))
        width = 0.8 / len(tipos)
        
        colores = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, tipo in enumerate(tipos):
            offset = width * i - (width * len(tipos) / 2)
            valores = df_año[tipo].values
            ax.bar(x + offset, valores, width, 
                  label=tipo.replace('_', ' ').title(),
                  color=colores[i % len(colores)], alpha=0.8)
        
        ax.set_xlabel('Mes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cantidad de vehículos', fontsize=12, fontweight='bold')
        ax.set_title(f'Frecuencias por Tipo de Vehículo - Año {año}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df_año['MES'].values, rotation=45, ha='right')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @output
    @render.ui
    def resumen_tipos():
        """Resumen estadístico"""
        año = int(input.año_visualizacion())
        tipos = list(input.tipos_vehiculos())
        
        if not tipos:
            return ui.p("Selecciona tipos de vehículos")
        
        df_año = df[df['AÑO'] == año]
        
        resumen_html = ""
        for tipo in tipos:
            total = df_año[tipo].sum()
            promedio = df_año[tipo].mean()
            maximo = df_año[tipo].max()
            
            resumen_html += f"""
            <div class="metric-card">
                <h5>{tipo.replace('_', ' ').title()}</h5>
                <p><strong>Total:</strong> {total:,.0f}</p>
                <p><strong>Promedio:</strong> {promedio:,.0f}</p>
                <p><strong>Máximo:</strong> {maximo:,.0f}</p>
            </div>
            """
        
        return ui.HTML(resumen_html)

# Crear app
app = App(app_ui, server)