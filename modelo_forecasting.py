import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class ForecastingModel:
    def __init__(self):
        self.models = {}
    
    def train_arima(self, series, vehicle_type):
        # Implementar ARIMA con división 80/20
        pass
    
    def train_prophet(self, df, vehicle_type):
        # Implementar Prophet
        pass
    
    def predict_next_month(self, vehicle_type):
        # Predecir junio 2025
        pass