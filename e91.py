import numpy as np
import random
import math # Needed for pi and sqrt

def run_e91_simulation(eavesdropping_active=False,
                       key_length=1000,
                       noise_level=0.5):
    """
    Simulates the E91 protocol (Ekert 1991), which uses Bell's Theorem
    (CHSH Statistic S) to guarantee key security.

    Security is verified by checking if the observed Bell value |S| is
    greater than the classical limit (2.0).

    :param eavesdropping_active: If True, entanglement is partially destroyed, 
                                 forcing |S| toward the classical limit (<= 2).
    :param key_length: Number of EPR pairs simulated.
    :param noise_level: Level of classical noise introduced (0 to 1).
    :return: A dictionary containing S_Calculated, Bell_Violation status, and key length.
    """

    # --- 1. Definition of Parameters and Bases ---
    
    # Angles of the measurement bases (in radians)
    ALICE_BASES_ANGLES = np.array([0, math.pi/4, math.pi/2])
    ALICE_BASES_NAMES = np.array(["a1", "a2", "a3"])
    
    BOB_BASES_ANGLES = np.array([math.pi/4, math.pi/2, 3*math.pi/4])
    BOB_BASES_NAMES = np.array(["b1", "b2", "b3"])
    
    S_QUANTUM_PREDICTION = -2 * math.sqrt(2) 
    S_CLASSIC_LIMIT = 2.0
    
    def quantum_correlation_E(theta_a, theta_b):
        """E(a, b) = -cos(theta_a - theta_b) for the Singlet state."""
        return -math.cos(theta_a - theta_b)
    
    def simulate_entangled_measurement(alice_angle, bob_angle, eavesdropping):
        E_ideal = quantum_correlation_E(alice_angle, bob_angle)
        
        # Simulates attack by reducing correlation towards classical behavior.
        E_actual = E_ideal * (1 - noise_level) if eavesdropping else E_ideal
        
        # P(same sign) = (1 + E_actual) / 2
        P_same_sign = (1 + E_actual) / 2 
        
        # Alice measures (+1 or -1)
        alice_result = random.choice([1, -1])
        
        # Bob measures based on correlation probability
        if random.random() < P_same_sign:
            bob_result = alice_result # Same sign
        else:
            bob_result = -alice_result # Opposite sign
        
        return alice_result, bob_result

    # --- 2. Generation of Data and Measurements ---
    
    # Indices 0, 1, 2 correspond to the three bases
    alice_base_indices = np.random.randint(0, 3, key_length)
    bob_base_indices = np.random.randint(0, 3, key_length)
    
    alice_results = np.zeros(key_length, dtype=int)
    bob_results = np.zeros(key_length, dtype=int)
    
    key_bases = np.empty((key_length, 2), dtype=object)
    
    for i in range(key_length):
        a_idx = alice_base_indices[i]
        b_idx = bob_base_indices[i]
        
        theta_a = ALICE_BASES_ANGLES[a_idx]
        theta_b = BOB_BASES_ANGLES[b_idx]
        
        a_res, b_res = simulate_entangled_measurement(theta_a, theta_b, eavesdropping_active)
        
        alice_results[i] = a_res
        bob_results[i] = b_res
        key_bases[i, 0] = ALICE_BASES_NAMES[a_idx]
        key_bases[i, 1] = BOB_BASES_NAMES[b_idx]
    
    # --- 3. Sifting for the Secret Key ---
    
    # Key bases are where anti-correlations are perfect: (a2, b1) and (a3, b2)
    key_indices = np.where(
        (key_bases[:, 0] == "a2") & (key_bases[:, 1] == "b1") | 
        (key_bases[:, 0] == "a3") & (key_bases[:, 1] == "b2")
    )[0]
    
    raw_key_alice = alice_results[key_indices]
    raw_key_bob_raw = bob_results[key_indices]
    
    # Bob flips his bit due to the expected anti-correlation in the singlet state
    raw_key_bob_final = -raw_key_bob_raw 
    sifted_key_length = len(raw_key_alice)
    
    # --- 4. Bell Test (Calculation of S) ---
    
    def calculate_E(a_name, b_name, key_bases, alice_results, bob_results):
        indices = np.where((key_bases[:, 0] == a_name) & (key_bases[:, 1] == b_name))[0]
        if len(indices) < 2: 
            return 0.0 # Return 0 if not enough data, though NA is safer, 0 allows summation
        
        # E = Average of the product of results (R_Alice * R_Bob)
        E_value = np.mean(alice_results[indices] * bob_results[indices])
        return E_value
    
    # CHSH Formula: S = E(a1, b1) - E(a1, b3) + E(a3, b1) + E(a3, b3)
    E_a1_b1 = calculate_E("a1", "b1", key_bases, alice_results, bob_results)
    E_a1_b3 = calculate_E("a1", "b3", key_bases, alice_results, bob_results)
    E_a3_b1 = calculate_E("a3", "b1", key_bases, alice_results, bob_results)
    E_a3_b3 = calculate_E("a3", "b3", key_bases, alice_results, bob_results)
    
    S_calculated = E_a1_b1 - E_a1_b3 + E_a3_b1 + E_a3_b3
    
    # --- 5. Result Verification ---
    
    bell_violation = abs(S_calculated) > S_CLASSIC_LIMIT
    
    # Calculate error rate in the final key for completeness
    errors_in_key = np.sum(raw_key_alice != raw_key_bob_final)
    error_rate_key = errors_in_key / sifted_key_length if sifted_key_length > 0 else 0.0

    result = {
        "S_Calculated": S_calculated,
        "S_Theoretical": S_QUANTUM_PREDICTION,
        "S_Classic_Limit": S_CLASSIC_LIMIT,
        "Bell_Violation": bell_violation,
        "Sifted_Key_Length": sifted_key_length,
        "Key_Error_Rate": error_rate_key,
        "Eavesdropping": eavesdropping_active
    }
    
    return result

if __name__ == '__main__':
    # --- Example Usage ---
    
    # Scenario 1: Secure Channel (No Eavesdropping)
    results_secure = run_e91_simulation(eavesdropping_active=False)
    print("\n--- Scenario 1: Secure Channel (No Eavesdropping) ---")
    print(f"S Calculated (CHSH): {results_secure['S_Calculated']:.4f}")
    print(f"Bell Violation (|S| > 2): {results_secure['Bell_Violation']}")
    print(f"Key Error Rate: {results_secure['Key_Error_Rate']:.4f}")
    
    if results_secure['Bell_Violation']:
        print("✅ Entanglement detected. Key is secure.")
    
    # Scenario 2: Attack that Breaks Entanglement (High Noise)
    results_attack = run_e91_simulation(eavesdropping_active=True, noise_level=0.8)
    print("\n--- Scenario 2: Attack (Entanglement Broken) ---")
    print(f"S Calculated (CHSH): {results_attack['S_Calculated']:.4f}")
    print(f"Bell Violation (|S| > 2): {results_attack['Bell_Violation']}")
    print(f"Key Error Rate: {results_attack['Key_Error_Rate']:.4f}")
    
    if not results_attack['Bell_Violation']:
        print("🚨 ALERTA: Bell inequality violated. Entanglement broken. Key discarded.")
