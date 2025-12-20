import numpy as np
from datetime import datetime
from skyfield.api import utc
from scheduler import QuantumSatelliteScheduler

def main():
    # Datos de ejemplo (Micius Satellite)
    line1 = "1 41731U 16051A   24350.52843431  .00000969  00000-0  16238-3 0  9997"
    line2 = "2 41731  97.4024 252.3661 0008272  97.6698 262.5358 15.22855523463832"
    
    print("--- Scenario: Ground Station Teide (Canary Islands) ---")
    scheduler = QuantumSatelliteScheduler(line1, line2, 28.3, -16.5, 2393)
    
    results = scheduler.find_optimal_windows(datetime.now(utc), 24)
    passes = np.where(results['mask'])[0]
    
    if len(passes) > 0:
        best_rate = np.max(results['rates'])
        print(f"Status: SUCCESS")
        print(f"Optimal Pass Duration: {len(passes)/2} minutes")
        print(f"Peak QKD Rate: {best_rate:.2f} bps")
        print(f"Total Potential Keys (10min): {int(best_rate * 600)} bits")
    else:
        print("Status: NO VISIBLE PASSES")
        print("Recommendation: Wait for next orbital cycle (~94 min).")

if __name__ == "__main__":
    main()
