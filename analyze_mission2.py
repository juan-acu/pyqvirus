import numpy as np
from datetime import datetime
from skyfield.api import utc
from scheduler import QuantumSatelliteScheduler

def identify_passes(mask_array, times_array):
    """Segmenta el array de visibilidad en pasadas individuales."""
    passes = []
    in_pass = False
    pass_start = None
    for i, is_visible in enumerate(mask_array):
        if is_visible and not in_pass:
            in_pass = True
            pass_start = i
        elif not is_visible and in_pass:
            in_pass = False
            passes.append((pass_start, i-1))
            pass_start = None
    if in_pass and pass_start is not None:
        passes.append((pass_start, len(mask_array)-1))
    return passes

def main():
    # 1. CONFIGURACIÓN
    l1 = "1 41731U 16051A   24350.52843431  .00000969  00000-0  16238-3 0  9997"
    l2 = "2 41731  97.4024 252.3661 0008272  97.6698 262.5358 15.22855523463832"
    scheduler = QuantumSatelliteScheduler(l1, l2, 28.3, -16.5, 2393)
    start_time = datetime.now(utc)
    
    # 2. EJECUCIÓN DEL SCHEDULER
    print(f"🚀 Iniciando análisis de misión QKD para las próximas 24h...")
    results = scheduler.find_optimal_windows(start_time, duration_hours=24)
    passes = identify_passes(results['mask'], results['times'])
    
    print(f"📡 Se han detectado {len(passes)} pasadas válidas (>30°).")

    # 3. DETALLE DE PASADAS
    total_keys_24h = 0
    for i, (start_idx, end_idx) in enumerate(passes, 1):
        pass_times = results['times'][start_idx:end_idx+1]
        pass_elevs = results['elevations'][start_idx:end_idx+1]
        pass_rates = results['rates'][start_idx:end_idx+1]
        
        duration_min = (end_idx - start_idx) / 2
        max_elev = np.max(pass_elevs)
        total_keys = np.sum(pass_rates) * 30 
        total_keys_24h += total_keys
        
        if max_elev > 70: color, label = "🟢", "EXCELENTE"
        elif max_elev > 50: color, label = "🟡", "BUENA"
        else: color, label = "🟠", "ACEPTABLE"

        print(f"\n{'-'*40}")
        print(f"PASADA #{i} | {color} {label}")
        print(f"{'-'*40}")
        print(f"  Ventana:  {pass_times[0].utc_strftime('%H:%M:%S')} - {pass_times[-1].utc_strftime('%H:%M:%S')} UTC")
        print(f"  Máx Elev: {max_elev:.1f}° | Claves: {total_keys:.0f} bits")

    # 4. INTELIGENCIA OPERATIVA (UPDATE)
    print("\n" + "="*60)
    print("🤖 RECOMENDACIONES DEL OPTIMIZADOR IA")
    print("="*60)

    if len(passes) == 0:
        print("❌ No hay pasadas disponibles. Recomendación: Stand-by 24h.")
    else:
        # Encontrar mejor pasada para prioridad
        best_pass_idx = np.argmax([np.max(results['elevations'][s:e+1]) for s, e in passes])
        best_s, _ = passes[best_pass_idx]
        best_time = results['times'][best_s]
        
        time_until = (best_time.utc_datetime() - datetime.now(utc)).total_seconds()
        
        print(f"🎯 PASADA ÓPTIMA: Pasada #{best_pass_idx + 1}")
        print(f"⏱️  Tiempo restante: {int(time_until//3600)}h {int((time_until%3600)//60)}m")
        
        print("\n💡 ACCIONES SUGERIDAS:")
        print("  1. Verificar meteorología (nubes < 30%)")
        print("  2. Calibrar tracking óptico y Pre-enfriar detectores")
        print("  3. Sincronizar relojes atómicos (< 1ns)")

    print("\n📊 PREDICCIÓN GLOBAL (24h):")
    print(f"  Total bits: {total_keys_24h:.0f} | Claves AES-256: {int(total_keys_24h/2048)}")
    print("="*60)
    print("✅ ANÁLISIS COMPLETO - Scheduler listo para operación")

if __name__ == "__main__":
    main()
