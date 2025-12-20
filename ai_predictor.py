import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

class TurbulencePredictorRNN:
    """
    Predictor basado en Deep Learning para pronóstico de Cn2.
    Utiliza una arquitectura LSTM Bidireccional para procesar series temporales
    meteorológicas y detectar ventanas óptimas para QKD.
    """
    def __init__(self, lookback_hours=6, forecast_hours=12):
        """
        Parameters:
        -----------
        lookback_hours : int (Horas de historia para el contexto)
        forecast_hours : int (Horas a predecir hacia el futuro)
        """
        self.lookback = lookback_hours * 60  # convertir a minutos
        self.forecast = forecast_hours * 60
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self):
        """Construye una arquitectura LSTM Bidireccional Profunda"""
        model = keras.Sequential([
            # Input: [batch, timesteps, features] -> (None, 360, 5)
            keras.layers.Input(shape=(self.lookback, 5)),
            
            # Capa 1: Captura patrones temporales en ambos sentidos
            keras.layers.Bidirectional(
                keras.layers.LSTM(128, return_sequences=True)
            ),
            keras.layers.Dropout(0.2),
            
            # Capa 2: Refinamiento de características secuenciales
            keras.layers.Bidirectional(
                keras.layers.LSTM(64, return_sequences=True)
            ),
            keras.layers.Dropout(0.2),
            
            # Capa 3: Reducción de dimensionalidad temporal
            keras.layers.Bidirectional(
                keras.layers.LSTM(32, return_sequences=False)
            ),
            keras.layers.Dropout(0.2),
            
            # Capas Densas para interpretación de ruido
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            
            # Output: Predicción continua de Cn2 (720 minutos)
            keras.layers.Dense(self.forecast, activation='relu')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model

    def prepare_data(self, weather_data):
        """
        Transforma un DataFrame meteorológico en tensores X, y para entrenamiento.
        Columnas esperadas: ['temperature', 'humidity', 'pressure', 'wind_speed', 'turbulence_cn2']
        """
        features = weather_data[['temperature', 'humidity', 'pressure', 
                                'wind_speed', 'turbulence_cn2']].values
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(len(features_scaled) - self.lookback - self.forecast):
            # X: Ventana de entrada (360 mins)
            X.append(features_scaled[i:i+self.lookback])
            # y: Target de turbulencia futura (720 mins)
            y.append(features_scaled[i+self.lookback:i+self.lookback+self.forecast, 4])
        
        return np.array(X), np.array(y)

    def predict_turbulence(self, recent_data):
        """
        Inferencia: Predice los próximos 720 minutos de Cn2 basándose en las últimas 6 horas.
        """
        # Normalizar y ajustar dimensiones para el modelo
        recent_scaled = self.scaler.transform(recent_data)
        recent_scaled = recent_scaled.reshape(1, self.lookback, 5)
        
        # Predicción (desactivando logs de TF para salida limpia)
        prediction_scaled = self.model.predict(recent_scaled, verbose=0)[0]
        
        # Des-normalizar usando el scaler original (columna index 4: Cn2)
        prediction = prediction_scaled * self.scaler.scale_[4] + self.scaler.mean_[4]
        return prediction

    def identify_optimal_windows(self, prediction, threshold_cn2=1.2e-14):
        """
        Algoritmo de segmentación para hallar periodos de estabilidad.
        Returns: [(inicio, fin, promedio_cn2), ...]
        """
        good_minutes = prediction < threshold_cn2
        windows = []
        start = None
        
        for i, is_good in enumerate(good_minutes):
            if is_good and start is None:
                start = i
            elif not is_good and start is not None:
                avg_cn2 = np.mean(prediction[start:i])
                windows.append((start, i, avg_cn2))
                start = None
        
        # Capturar ventana si queda abierta al final del array
        if start is not None:
            avg_cn2 = np.mean(prediction[start:])
            windows.append((start, len(prediction), avg_cn2))
            
        return windows
