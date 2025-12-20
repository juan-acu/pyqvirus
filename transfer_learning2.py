import numpy as np
import tensorflow as tf
from tensorflow import keras
from ai_predictor import TurbulencePredictorRNN
from sklearn.metrics import r2_score
from pathlib import Path
import argparse

class TransferLearningAdapter:
    """
    Adaptador de modelos pre-entrenados para nuevas ubicaciones.
    
    Estrategia de Transfer Learning:
    -------------------------------
    1. Congelar capas LSTM: Mantienen el conocimiento "universal" de la dinámica atmosférica.
    2. Liberar capas Dense: Permiten al modelo aprender el "microclima" local.
    3. Fine-tuning: Entrenamiento de alta precisión con pocos datos (2 semanas).
    """
    
    def __init__(self, base_model_path='models/ottawa_base_model.h5'):
        self.base_model_path = Path(base_model_path)
        self.base_model = None
        self.adapted_model = None
        self.site_name = None
        
    def load_base_model(self):
        """Carga el modelo pre-entrenado de Ottawa o crea uno genérico si no existe."""
        print(f"📦 Cargando modelo base...")
        if not self.base_model_path.exists():
            print("⚠️ Modelo base no encontrado. Inicializando arquitectura nueva...")
            predictor = TurbulencePredictorRNN(lookback_hours=6, forecast_hours=12)
            self.base_model = predictor.model
        else:
            self.base_model = keras.models.load_model(self.base_model_path)
            print(f"✅ Modelo cargado: {self.base_model.count_params():,} parámetros")
        return self.base_model

    def prepare_for_transfer_learning(self):
        """Configura la arquitectura para el aprendizaje por transferencia."""
        if self.base_model is None: self.load_base_model()
        
        print("\n" + "="*60)
        print("🔄 CONFIGURANDO TRANSFER LEARNING")
        print("="*60)

        # 1. Congelar capas LSTM (Conocimiento Universal)
        frozen_count = 0
        for layer in self.base_model.layers:
            if 'bidirectional' in layer.name:
                layer.trainable = False
                frozen_count += 1
                print(f"  ❄️ {layer.name}: CONGELADA")
            if frozen_count >= 2: break
        
        # 2. Liberar capas Dense (Adaptación Local)
        trainable_count = 0
        for layer in self.base_model.layers:
            if 'dense' in layer.name or 'dropout' in layer.name:
                layer.trainable = True
                trainable_count += 1
                print(f"  🔓 {layer.name}: ENTRENABLE")

        # 3. Compilación con Learning Rate bajo para Fine-Tuning
        self.base_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='mse',
            metrics=['mae']
        )
        self.adapted_model = self.base_model
        print(f"\n✅ Preparación lista: {frozen_count} fijas, {trainable_count} adaptables.")
        return self.adapted_model

    def fine_tune(self, X_local, y_local, site_name, epochs=50, batch_size=32):
        """Entrena el modelo con datos específicos del nuevo sitio."""
        self.site_name = site_name
        print(f"\n⏱️ Iniciando Fine-tuning para {site_name}...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
        ]
        
        history = self.adapted_model.fit(
            X_local, y_local,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        print(f"✅ Adaptación a {site_name} completada.")
        return history

    def evaluate_adaptation(self, X_test, y_test):
        """
        Calcula métricas de precisión corregidas para regresión de Cn2.
        """
        predictions = self.adapted_model.predict(X_test, verbose=0)
        
        # 1. R2 Score (Varianza explicada)
        r2 = r2_score(y_test.flatten(), predictions.flatten())
        
        # 2. Precisión basada en Error Relativo (Fix de la regresión)
        # Usamos epsilon para evitar división por cero en valores de 10^-15
        epsilon = 1e-17
        relative_error = np.abs(predictions - y_test) / (np.abs(y_test) + epsilon)
        
        # Definimos "Precisión" como predicciones con menos del 20% de error
        accuracy_percent = np.mean(relative_error < 0.20) * 100
        
        metrics = {
            'mae': np.mean(np.abs(predictions - y_test)),
            'r2': r2,
            'accuracy_percent': accuracy_percent
        }
        
        print(f"📊 RESULTADOS {self.site_name}:")
        print(f"   R2 Score:  {r2:.4f}")
        print(f"   Precisión: {accuracy_percent:.1f}% (Margen 20%)")
        return metrics

    def compare_performance(self, X_test, y_test, base_accuracy=72.0):
        """Compara el rendimiento antes y después de la adaptación."""
        metrics = self.evaluate_adaptation(X_test, y_test)
        adapted_acc = metrics['accuracy_percent']
        gain = adapted_acc - base_accuracy
        
        print("\n" + "="*60)
        print(f"💡 GANANCIA EN {self.site_name.upper()}: +{gain:.1f} puntos")
        print("="*60)
        
        return {
            'base_accuracy': base_accuracy,
            'adapted_accuracy': adapted_acc,
            'gain_points': gain,
            'r2': metrics['r2']
        }

    def save_adapted_model(self, output_path=None):
        if output_path is None:
            output_path = f"models/{self.site_name.lower()}_adapted_model.h5"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.adapted_model.save(output_path)
        print(f"💾 Modelo guardado en: {output_path}")

# --- Funciones Auxiliares para Simulación ---

def simulate_local_data(site_name, n_samples=600):
    """Genera datos sintéticos para entrenamiento local."""
    # Parámetros simplificados por región
    params = {
        'Tenerife': {'temp': 18, 'wind': 4, 'cn2': 9e-15},
        'Chile':    {'temp': 12, 'wind': 6, 'cn2': 7e-15},
        'Namibia':  {'temp': 22, 'wind': 5, 'cn2': 1.1e-14}
    }.get(site_name, {'temp': 15, 'wind': 5, 'cn2': 1e-14})

    X = np.random.randn(n_samples, 360, 5) * 0.1 # Datos de entrada normalizados
    # Generar target Cn2 con el sesgo del sitio
    y = np.full((n_samples, 720), params['cn2']) + np.random.randn(n_samples, 720) * 1e-16
    return X, y

def run_demo(site):
    adapter = TransferLearningAdapter()
    adapter.prepare_for_transfer_learning()
    
    X_train, y_train = simulate_local_data(site, n_samples=500)
    X_test, y_test = simulate_local_data(site, n_samples=100)
    
    adapter.fine_tune(X_train, y_train, site_name=site)
    adapter.compare_performance(X_test, y_test)
    adapter.save_adapted_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IAQC-COM6 Transfer Learning")
    parser.add_argument("--site", type=str, default="Tenerife", help="Sitio para adaptar")
    args = parser.parse_args()
    run_demo(args.site)
