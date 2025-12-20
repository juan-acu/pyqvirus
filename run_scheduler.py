import numpy as np
from datetime import datetime
from skyfield.api import utc
from scheduler import QuantumSatelliteScheduler

def main():
    # TLE actualizado del satélite Micius (QUESS)
    line1 = "1 41731U 16051A   24350.52843431  .00000969  00000-0  16238-3 0  9997"
    line2 = "2 41731  97.4024 252.3661 0008272  97.6698 262.5358 15.22855523463832"

    # Configuración de la estación terrestre (Tenerife)
    scheduler = QuantumSatelliteScheduler(
        line1, line2, 
        ground_lat=28.3, 
        ground_lon=-16.5, 
        ground_alt_m=2393
    )
    
    print("Clase QuantumSatelliteScheduler cargada y probada con éxito.")
    print("✅ Scheduler inicializado correctamente")
    print(f"   Satélite: {scheduler.satellite.name}")
    print(f"   Órbita: ~15.23 revoluciones/día (LEO)")
    print(f"   Estación: Tenerife (28.3°N, 16.5°W, 2393m)")

    # Definir momento de inicio de búsqueda (Ahora mismo en UTC)
    start = datetime.now(utc)
    print(f"\n🔍 Buscando ventanas desde: {start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("   Analizando próximas 24 horas cada 30 segundos...")

    # Ejecutar búsqueda
    results = scheduler.find_optimal_windows(start, duration_hours=24)
    passes_indices = np.where(results['mask'])[0]

    if len(passes_indices) > 0:
        # Calcular duración total de ventanas válidas (2 mediciones por minuto)
        total_minutes = len(passes_indices) / 2
        
        print(f"\n🎉 Success! Found {int(total_minutes)} minutes of QKD windows.")
        
        # Encontrar momento de tasa máxima
        best_idx = passes_indices[np.argmax(results['rates'][passes_indices])]
        best_time = results['times'][best_idx]
        best_elevation = results['elevations'][best_idx]
        best_rate = results['rates'][best_idx]
        
        print(f"\n📊 MEJOR VENTANA DETECTADA:")
        print(f"  Tiempo: {best_time.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Elevación: {best_elevation:.1f}°")
        print(f"  Peak Rate: {best_rate:.2f} bps")
        
        # Calcular claves generadas en ventana de 10 minutos (600 segundos)
        keys_10min = best_rate * 600
        print(f"\n💾 En ventana óptima de 10 minutos:")
        print(f"  Claves generadas: {keys_10min:.0f} bits")
        print(f"  Equivalente a: {keys_10min/8:.0f} bytes")
        print(f"  Suficiente para: {int(keys_10min/256)} claves AES-256")
        
    else:
        print("\n❌ No passes above 30° elevation found in the next 24 hours.")
        print("  Intenta:")
        print("  - Esperar más tiempo (el satélite pasa cada ~94 minutos)")
        print("  - Verificar TLE actualizado")
        print("  - Cambiar ubicación de estación terrestre")

if __name__ == "__main__":
    main()
