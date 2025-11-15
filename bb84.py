import numpy as np
import random
# Removed: from typing import Dict, Any 

def run_bb84_simulation(eavesdropping_active=False,
                       key_length=1000,
                       test_ratio=0.2):
    """
    Simulates the BB84 protocol for quantum key distribution (QKD).

    Models encoding, transmission, and measurement of qubits, illustrating
    the difference in QBER between an ideal channel and an eavesdropped one.

    :param eavesdropping_active: If True, Eve performs a measure-and-resend attack.
    :param key_length: The total number of photons simulated.
    :param test_ratio: Proportion of sifted bits used for QBER calculation.
    :return: A dictionary containing the QBER, key lengths, and status.
    """

    # --- 1. Alice's Preparation ---
    
    # R's sample(0:1, length, replace=TRUE) is np.random.randint(0, 2, length)
    alice_bits = np.random.randint(0, 2, key_length)
    alice_bases = np.random.randint(0, 2, key_length)
    original_alice_bits = alice_bits.copy()
    
    # R's mapply(alice_encode, ...) is a list of dictionaries (photons)
    photons_sent = []
    for bit, base in zip(alice_bits, alice_bases):
        photons_sent.append({
            'encoded_bit': bit, 
            'encoding_base': base
        })

    # --- 2. Eavesdropping (Eve's Attack) ---
    if eavesdropping_active:
        eve_bases = np.random.randint(0, 2, key_length)
        for i in range(key_length):
            a_info = photons_sent[i]
            eve_base = eve_bases[i]
            
            # Quantum Rule Simulation: If bases match, Eve gets the bit. 
            if a_info['encoding_base'] == eve_base:
                eve_bit = a_info['encoded_bit']
            else:
                # If not, the result is random (sample(0:1, 1) in R).
                eve_bit = random.choice([0, 1])
            
            # Eve resends a new photon based on her measurement and base
            photons_sent[i] = {
                'encoded_bit': eve_bit, 
                'encoding_base': eve_base
            }

    # --- 3. Bob's Measurement ---
    bob_bases = np.random.randint(0, 2, key_length)
    bob_bits = np.full(key_length, -1, dtype=int) # np.full replaces rep(NA, length)

    for i in range(key_length):
        photon = photons_sent[i]
        bob_base = bob_bases[i]
        
        # Quantum Rule Simulation (Bob): Same logic as Eve's measurement
        if photon['encoding_base'] == bob_base:
            bob_bits[i] = photon['encoded_bit']
        else:
            bob_bits[i] = random.choice([0, 1])
    
    # --- 4. Sifting (Basis Reconciliation) ---
    
    # R's which(...) is np.where(...)
    matching_indices = np.where(alice_bases == bob_bases)[0]
    
    raw_key_alice = original_alice_bits[matching_indices]
    raw_key_bob = bob_bits[matching_indices]
    
    sifted_length = len(raw_key_alice)
    test_size = int(sifted_length * test_ratio)
    
    # R's sample(seq_len(...), size) is random.sample()
    if sifted_length > 0 and test_size > 0:
        test_indices = random.sample(range(sifted_length), test_size)
    else:
        # Handle cases where sifting fails to yield a sufficient key
        return {
            "QBER": 0.0,
            "Sifted_Length": 0,
            "Final_Key_Length": 0,
            "Eavesdropping": eavesdropping_active
        }

    # --- 5. Error Estimation (QBER) ---
    
    # R's sum(A != B) is np.sum(A != B)
    errors = np.sum(raw_key_alice[test_indices] != raw_key_bob[test_indices])
    error_rate = errors / test_size
    
    # R's setdiff(seq_len(...), test_indices)
    all_indices = set(range(sifted_length))
    secret_indices = list(all_indices - set(test_indices))
    final_key_length = len(secret_indices)

    # --- 6. Return Result ---
    
    result = {
        "QBER": error_rate,
        "Sifted_Length": sifted_length,
        "Final_Key_Length": final_key_length,
        "Eavesdropping": eavesdropping_active
    }
    
    return result

if __name__ == '__main__':
    # --- Example Usage ---
    
    # Scenario 1: Perfect Channel (No Eavesdropping)
    results_no_eve = run_bb84_simulation(eavesdropping_active=False,key_length=10,test_ratio=0.3)
    print("\n--- Scenario 1: No Eavesdropping ---")
    print(f"QBER: {results_no_eve['QBER']:.4f}")
    print(f"Final Key Length: {results_no_eve['Final_Key_Length']}")
    
    # Scenario 2: Measure-and-Resend Attack
    results_with_eve = run_bb84_simulation(eavesdropping_active=True)
    print("\n--- Scenario 2: With Eavesdropping ---")
    print(f"QBER: {results_with_eve['QBER']:.4f}")
    print(f"Final Key Length: {results_with_eve['Final_Key_Length']}")
    
    if results_with_eve['QBER'] > 0.15:
        print("🚨 High QBER detected. Key discarded.")
