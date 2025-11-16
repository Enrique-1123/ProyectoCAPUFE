# 🚗 Dashboard CAPUFE - Forecasting de Vehículos

## 📊 Descripción
Dashboard interactivo para análisis y pronóstico de movimientos vehiculares en la red CAPUFE usando Series de Tiempo.

## 🎯 Objetivo
Aplicar técnicas de analítica y visualización de datos con algoritmo de aprendizaje supervisado predictivo (Series de Tiempo) para generar pronósticos.

## ✨ Características
- **Pronósticos** con modelos de series de tiempo
- **Análisis exploratorio** interactivo
- **Respuestas a cuestionamientos** específicos del proyecto
- **Visualizaciones** profesionales y métricas de evaluación

## 🛠️ Tecnologías Utilizadas
- Python + Shiny Framework
- Pandas, NumPy, Matplotlib
- Scikit-learn para métricas
- Series de Tiempo (Forecasting)

## 🚀 Instalación y Ejecución

### Prerrequisitos
- Python 3.8+
- Git

### Pasos para ejecutar localmente
```bash
# Clonar el repositorio
git clone https://github.com/tuusuario/capufe-dashboard.git
cd capufe-dashboard

# Crear entorno virtual (opcional pero recomendado)
python -m venv capufe_env
source capufe_env/bin/activate  # En Windows: capufe_env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
shiny run scripts/app.py