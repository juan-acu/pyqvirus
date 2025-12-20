import numpy as np
from datetime import datetime, timedelta
from skyfield.api import load, wgs84, EarthSatellite, utc

class QuantumSatelliteScheduler:
    def __init__(self, tle_line1, tle_line2, ground_lat, ground_lon, ground_alt_m=0):
        """
        Scheduler para enlaces QKD satelitales usando skyfield.
        """
        self.ts = load.timescale()
        self.satellite = EarthSatellite(tle_line1, tle_line2, 'QUESS', self.ts)
        self.ground_station = wgs84.latlon(ground_lat, ground_lon, elevation_m=ground_alt_m)
     
    def find_optimal_windows(self, start_datetime, duration_hours):
        """Encuentra ventanas óptimas de comunicación (Elevación > 30°)."""
        t0 = self.ts.from_datetime(start_datetime)
        t1 = self.ts.from_datetime(start_datetime + timedelta(hours=duration_hours))
        
        times = self.ts.linspace(t0, t1, int(duration_hours * 120))
        difference = self.satellite - self.ground_station
        topocentric = difference.at(times)
        alt, az, distance = topocentric.altaz()
        
        valid_mask = alt.degrees > 30
        qkd_rates = self.estimate_qkd_rate(alt.degrees, distance.km)
        
        return {
            'times': times,
            'elevations': alt.degrees,
            'rates': qkd_rates,
            'mask': valid_mask
        }
     
    def estimate_qkd_rate(self, elevation_deg, distance_km):
        """Modelo de tasa QKD basado en pérdida atmosférica y distancia."""
        elevation_factor = np.maximum(0, (elevation_deg - 30) / 60)
        distance_factor = np.exp(-(distance_km - 500) / 500)
        return elevation_factor * distance_factor * 1200
