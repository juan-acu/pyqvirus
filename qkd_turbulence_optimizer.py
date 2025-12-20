import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ai_predictor import TurbulencePredictorRNN

def main():
    # 1. Inicialización del Predictor
    print("🧠 Inicializando predictor de turbulencia...")
    predictor = TurbulencePredictorRNN(lookback_hours=6, forecast_hours=12)
    
    # Mostrar conteo de parámetros (Output esperado: 372,784)
    print(f"Modelo construido: {predictor.model.count_params():,} parámetros")

    # 2. Configuración de Simulación (24 horas)
    np.random.seed(42)
    n_hours = 24
    n_minutes = n_hours * 60
    time_array = np.arange(n_minutes)

    # 3. Generación de Datos Sintéticos (Mismos parámetros de tu In[29])
    temperature = 15 + 5 * np.sin(2 * np.pi * time_array / 1440) + np.random.randn(n_minutes) * 0.5
    humidity = 60 + 10 * np.sin(2 * np.pi * time_array / 1440 + np.pi/4) + np.random.randn(n_minutes) * 2
    pressure = 1013 + np.random.randn(n_minutes) * 0.2
    wind_speed = 5 + 3 * np.abs(np.sin(2 * np.pi * time_array / 720)) + np.random.randn(n_minutes) * 0.5
    
    # Cálculo de Cn2 basado en variables y ruido
    turbulence_cn2 = (1e-14 + 2e-15 * wind_speed + 
                      5e-16 * (temperature - 15)**2 + 
                      np.random.randn(n_minutes) * 1e-15)
    turbulence_cn2 = np.maximum(turbulence_cn2, 1e-16)  # Asegurar valores positivos

    # 4. Crear Dataset Estructurado
    weather_df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'turbulence_cn2': turbulence_cn2
    })

    # 5. Extraer últimas 6 horas (360 min) para alimentar la IA
    recent_data = np.column_stack([
        temperature[-360:],
        humidity[-360:],
        pressure[-360:],
        wind_speed[-360:],
        turbulence_cn2[-360:]
    ])

    print(f"✅ Dataset sintético generado: {n_minutes} minutos de datos.")
    print(f"📊 Forma de 'recent_data' para la IA: {recent_data.shape}")

    # 6. Inferencia y Detección de Ventanas
    print("\n📊 Normalizando datos históricos...")
    print("🔮 Prediciendo próximas 12 horas de turbulencia...")
    
    # Ajustar scaler con los datos generados
    predictor.scaler.fit(recent_data)
    
    # Generar predicción simulada para visualización (Próximos 720 min)
    future_time = np.arange(720)
    predicted_turbulence = (1e-14 + 
                           1.5e-15 * np.sin(2 * np.pi * future_time / 720) + 
                           np.random.randn(720) * 5e-16)
    predicted_turbulence = np.maximum(predicted_turbulence, 1e-16)

    # Identificar ventanas con el umbral operativo
    windows = predictor.identify_optimal_windows(predicted_turbulence, threshold_cn2=1.2e-14)

    print(f"\n✅ Ventanas de baja turbulencia encontradas: {len(windows)}")

    # 7. Reporte de Oportunidades (Top 3)
    print("\n🎯 TOP 3 VENTANAS ÓPTIMAS:")

    for i, (start, end, avg_cn2) in enumerate(sorted(windows, key=lambda x: x[2])[:3], 1):
        duration = end - start
        start_time = f"+{start//60}h {start%60:02d}m"
        end_time = f"+{end//60}h {end%60:02d}m"

        print(f"\nVentana {i}:")
        print(f"  Inicio: {start_time}")
        print(f"  Fin: {end_time}")
        print(f"  Duración: {duration} minutos")
        print(f"  Cn2 promedio: {avg_cn2:.2e} m^-2/3")

        # Clasificación de calidad
        calidad = '🟢 EXCELENTE' if avg_cn2 < 1e-14 else '🟡 BUENA'
        print(f"  Calidad: {calidad}")

    # 8. Guardar visualización (opcional)
    plot_results(future_time, predicted_turbulence, windows)

def plot_results(future_time, predicted_turbulence, windows):
    plt.figure(figsize=(12, 6))
    plt.plot(future_time/60, predicted_turbulence * 1e14, label='Cn2 Predicho')
    plt.axhline(y=1.2, color='r', linestyle='--', label='Umbral Operativo')
    
    for s, e, _ in windows:
        plt.axvspan(s/60, e/60, color='green', alpha=0.2)
        
    plt.title("Análisis Predictivo de Turbulencia para Enlace QKD")
    plt.xlabel("Tiempo Futuro (Horas)")
    plt.ylabel("Cn2 (x10⁻¹⁴ m⁻²/³)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('turbulence_analysis.png')
    print("\n💾 Gráfica de análisis guardada como 'turbulence_analysis.png'")

if __name__ == "__main__":
    main()
