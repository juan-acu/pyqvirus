import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

class TurbulencePredictorRNN:
    """
    Predictor basado en Deep Learning para pronóstico de Cn2.
    Utiliza una arquitectura LSTM Bidireccional para procesar series temporales.
    """
    def __init__(self, lookback_hours=6, forecast_hours=12):
        self.lookback = lookback_hours * 60
        self.forecast = forecast_hours * 60
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self):
        # Arquitectura optimizada para señales ruidosas atmosféricas
        model = keras.Sequential([
            keras.layers.Input(shape=(self.lookback, 5)),
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            keras.layers.Dropout(0.2),
            keras.layers.Bidirectional(keras.layers.LSTM(64)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.forecast, activation='relu')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict_turbulence(self, recent_data):
        # Normalización y predicción
        recent_scaled = self.scaler.transform(recent_data).reshape(1, self.lookback, 5)
        prediction_scaled = self.model.predict(recent_scaled, verbose=0)[0]
        # Inversión de escala para la columna Cn2 (índice 4)
        return prediction_scaled * self.scaler.scale_[4] + self.scaler.mean_[4]
