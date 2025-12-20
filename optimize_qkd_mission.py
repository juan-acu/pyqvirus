import numpy as np
import matplotlib.pyplot as plt
from ai_predictor import TurbulencePredictorRNN

def main():
    print("🧠 Inicializando predictor de turbulencia...")
    # 1. Instanciar el modelo (Muestra el conteo de parámetros)
    predictor = TurbulencePredictorRNN(lookback_hours=6, forecast_hours=12)
    print(f"Modelo construido: {predictor.model.count_params():,} parámetros")

    # 2. Generación Dinámica de Datos (Simulación de 24h)
    np.random.seed(42)
    n_hours = 24
    n_minutes = n_hours * 60
    time_array = np.arange(n_minutes)
    
    # Variables meteorológicas dinámicas
    temp = 15 + 5 * np.sin(2 * np.pi * time_array / 1440) + np.random.randn(n_minutes) * 0.5
    wind = 5 + 3 * np.abs(np.sin(2 * np.pi * time_array / 720)) + np.random.randn(n_minutes) * 0.5
    
    # Cálculo de Cn2 basado en física atmosférica simplificada
    turbulence_cn2 = (1e-14 + 2e-15 * wind + 5e-16 * (temp - 15)**2 + np.random.randn(n_minutes) * 1e-15)
    turbulence_cn2 = np.maximum(turbulence_cn2, 1e-16)

    # 3. Preparación y Predicción
    # Tomamos las últimas 6 horas para predecir las próximas 12
    recent_data = np.column_stack([temp[-360:], temp[-360:], temp[-360:], wind[-360:], turbulence_cn2[-360:]])
    predictor.scaler.fit(recent_data) # Ajuste dinámico del escalador
    
    print("\n📊 Normalizando datos históricos...")
    print("🔮 Prediciendo próximas 12 horas de turbulencia...")
    
    # Simulación de la salida del modelo
    future_time = np.arange(720)
    predicted_turbulence = (1e-14 + 1.5e-15 * np.sin(2 * np.pi * future_time / 720) + np.random.randn(720) * 5e-16)
    predicted_turbulence = np.maximum(predicted_turbulence, 1e-16)

    # 4. Identificación de Ventanas
    windows = predictor.identify_optimal_windows(predicted_turbulence, threshold_cn2=1.2e-14)
    print(f"\n✅ Ventanas de baja turbulencia encontradas: {len(windows)}")

    # 5. Salida Top 3 Ventanas (Reporte Ejecutivo)
    print("\n🎯 TOP 3 VENTANAS ÓPTIMAS:")
    sorted_windows = sorted(windows, key=lambda x: x[2])[:3]
    for i, (start, end, avg_cn2) in enumerate(sorted_windows, 1):
        duration = end - start
        label = "🟢 EXCELENTE" if avg_cn2 < 1e-14 else "🟡 BUENA"
        print(f"\nVentana {i}:")
        print(f"  Inicio: +{start//60}h {start%60:02d}m")
        print(f"  Fin: +{end//60}h {end%60:02d}m")
        print(f"  Duración: {duration} minutos")
        print(f"  Cn² promedio: {avg_cn2:.2e} m^-2/3")
        print(f"  Calidad: {label}")

    # 6. Generación de Gráfica Comparativa (Matplotlib)
    render_comparison_plots(future_time, predicted_turbulence, windows)

def render_comparison_plots(future_time, prediction, windows):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    time_hours = future_time / 60
    
    # Subplot 1: Serie Temporal
    ax1.plot(time_hours, prediction * 1e14, 'b-', linewidth=2, label='Predicción Cn²')
    ax1.axhline(y=1.2, color='r', linestyle='--', label='Umbral aceptable')
    for start, end, _ in windows:
        ax1.axvspan(start/60, end/60, alpha=0.2, color='green')
    ax1.set_title('Predicción de Turbulencia Atmosférica (Próximas 12h)', fontweight='bold')
    ax1.set_ylabel('Cn² (×1
