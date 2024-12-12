#i realy hope you get me some Donation for the Quantum project_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
import cupy as cp  # CUDA-accelerated operations using CuPy
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.controlflow.break_loop import BreakLoopPlaceholder
from qiskit.circuit.library import ZGate, MCXGate
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit.primitives import SamplerResult
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from collections import Counter
from Crypto.Hash import RIPEMD160, SHA256  # Import from pycryptodome
from ecdsa import SigningKey, SECP256k1
from qiskit.quantum_info import Statevector
from bitarray import bitarray
from qiskit_aer.primitives import SamplerV2  # for simulator
from qiskit_ibm_runtime import SamplerV2 as real_sampler  # for hardware
from qiskit_aer import AerSimulator, Aer
import random
import time
import hashlib
import base58
import numpy as np
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Operator
import math

# Load IBMQ account using QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel='ibm_quantum',
    token='_REPLACE_WITH_YOUR_IBM_QUANTUM_TOKEN_RIGHT_HERE_',  # Replace with your actual token
    instance='ibm-q/open/main',
    overwrite=True,
    set_as_default=True
)

# Load the "open" credentials
service = QiskitRuntimeService()

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

from qiskit.circuit.library import QFT

def apply_qft(circuit, num_qubits):
    """
    Applies the Quantum Fourier Transform (QFT) on the first `num_qubits` qubits of the circuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit to which QFT is applied.
        num_qubits (int): The number of qubits to include in the QFT.
    """
    # Append the QFT library object for validation
    qft = QFT(num_qubits)
    circuit.append(qft, range(num_qubits))
    
    # Manual implementation of QFT for additional customization
    for j in range(num_qubits):
        circuit.h(j)
        for k in range(j + 1, num_qubits):
            circuit.cp(np.pi / (2 ** (k - j)), j, k)
    
    # Swap operations to reverse the order of qubits
    for j in range(num_qubits // 2):
        circuit.swap(j, num_qubits - j - 1)

# Function to convert private key to compressed Bitcoin address
def private_key_to_compressed_address(private_key_hex):
    print(f"Converting private key {private_key_hex} to Bitcoin address...")
    private_key_bytes = bytes.fromhex(private_key_hex)
    sk = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
    vk = sk.verifying_key
    public_key_bytes = vk.to_string()
    x_coord = public_key_bytes[:32]
    y_coord = public_key_bytes[32:]
    prefix = b'\x02' if int.from_bytes(y_coord, 'big') % 2 == 0 else b'\x03'
    compressed_public_key = prefix + x_coord

    sha256_pk = hashlib.sha256(compressed_public_key).digest()
    ripemd160 = RIPEMD160.new()  # Using Cryptodome's RIPEMD160
    ripemd160.update(sha256_pk)
    hashed_public_key = ripemd160.digest()

    network_byte = b'\x00' + hashed_public_key
    sha256_first = hashlib.sha256(network_byte).digest()
    sha256_second = hashlib.sha256(sha256_first).digest()
    checksum = sha256_second[:4]

    binary_address = network_byte + checksum
    bitcoin_address = base58.b58encode(binary_address).decode('utf-8')
    print(f"Generated Bitcoin address: {bitcoin_address}")
    return bitcoin_address     

# Function to convert public key to Public-Key-Hash (SHA256 -> RIPEMD160)
def public_key_to_public_key_hash(public_key_hex):    
    try:
        # Step 1: Perform SHA-256 on the public key
        sha256_hash = SHA256.new(bytes.fromhex(public_key_hex)).digest()
        
        # Step 2: Perform RIPEMD-160 on the result of SHA-256 using Cryptodome
        ripemd160 = RIPEMD160.new()  # Using Cryptodome's RIPEMD160
        ripemd160.update(sha256_hash)
        ripemd160_hash = ripemd160.digest()
        
        # Return the Public-Key-Hash in hexadecimal format
        return ripemd160_hash.hex()
    
    except ValueError as e:
        raise ValueError(f"Invalid input for public key hex: {e}")    

# New oracle that marks the target state for Grover's algorithm
def grover_oracle(circuit, private_key_qubits, public_key_x, g_x, g_y, p, ancilla):
    """Oracle for Grover's algorithm that checks if the current private key qubits 
    match the target public key via scalar multiplication."""
    
    circuit.barrier()
    keyspace_size = 0x1fffffffffffffffffffffffffffffff - 0x10000000000000000000000000000000 + 1
    num_qubits = 125
    # Removed the line that measures the qubits and calculates k here.
    # We will handle this after all iterations are complete.
    target_state = random.randint(0, keyspace_size - 1)  # Random target state within the keyspace
    target_state = random.randint(0, 2**num_qubits - 1)  # Pass the integer directly
    target_state = random.randint(0x10000000000000000000000000000000, 0x1fffffffffffffffffffffffffffffff)  # Random integer, not a binary string
    target_state_bin = format(target_state, f'0{num_qubits}b')  # Convert to binary string
    computed_x, _ = scalar_multiplication(target_state, g_x, g_y, p)

    # Flip the ancilla qubit if the computed x-coordinate matches the public key's x-coordinate
    if computed_x == public_key_x:
        circuit.x(ancilla)

    circuit.barrier()

# Supporting functions for elliptic curve operations
def mod_inverse(a, p):
    """Modular inverse function for elliptic curve operations."""
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % p, p
    while low > 1:
        ratio = high // low
        nm, new_low = hm - lm * ratio, high - low
        lm, low, hm, high = nm, new_low, lm, low
    return lm % p

def point_addition(x1, y1, x2, y2, p):
    """Elliptic curve point addition."""
    if x1 == x2 and y1 == y2:
        return point_doubling(x1, y1, p)
    lam = ((y2 - y1) * mod_inverse(x2 - x1, p)) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return x3, y3

def point_doubling(x1, y1, p):
    """Elliptic curve point doubling."""
    lam = ((3 * x1 * x1) * mod_inverse(2 * y1, p)) % p
    x3 = (lam * lam - 2 * x1) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return x3, y3

def scalar_multiplication(k, x, y, p):
    """Elliptic curve scalar multiplication."""
    x_res, y_res = x, y
    k_bin = bin(k)[2:]  # Convert the scalar k into binary representation
    for bit in k_bin[1:]:
        x_res, y_res = point_doubling(x_res, y_res, p)
        if bit == '1':
            x_res, y_res = point_addition(x_res, y_res, x, y, p)
    return x_res, y_res

def create_oracle_with_ancilla(num_qubits, target_state):
    """Creates an oracle circuit that marks the target state using an ancillary qubit."""
    oracle_circuit = QuantumCircuit(num_qubits + 1)  # +1 for ancilla

    # Initialize the ancilla qubit to |1⟩
    oracle_circuit.x(num_qubits)  # Set ancilla to |1⟩

    # Flip the bits to create the target state
    for i, bit in enumerate(bin(target_state)[2:].zfill(num_qubits)):
        if bit == '0':
            oracle_circuit.x(i)

    # Apply a controlled Z (CZ) gate to mark the target state
    oracle_circuit.cz(num_qubits, num_qubits - 1)  # Control on ancilla, target on last qubit

    # Flip back the bits to restore original state
    for i, bit in enumerate(bin(target_state)[2:].zfill(num_qubits)):
        if bit == '0':
            oracle_circuit.x(i)

    # Reset the ancilla qubit to |0⟩
    oracle_circuit.x(num_qubits)  # Reset the ancilla qubit

    return oracle_circuit

# Create Grover diffusion operator with CUDA-enabled matrix steps
def create_diffusion_operator_gpu(num_qubits):
    """Creates the Grover diffusion operator with CUDA-enabled steps."""
    qc = QuantumCircuit(num_qubits)

    # Step 1: Apply Hadamard gates
    qc.h(range(num_qubits))

    # Step 2: Apply X gates
    qc.x(range(num_qubits))

    # Step 3: Apply a Hadamard to the last qubit
    qc.h(num_qubits - 1)

    # Step 4: Apply multi-controlled X gate (using CUDA-accelerated matrix operations)
    mcx = MCXGate(num_qubits - 1)
    mcx_gpu = cp.asarray(mcx.to_matrix())  # Convert to CUDA-enabled matrix for GPU
    qc.append(mcx, list(range(num_qubits - 1)) + [num_qubits - 1])

    # Step 5: Apply Hadamard to the last qubit again
    qc.h(num_qubits - 1)

    # Step 6: Apply X gates again
    qc.x(range(num_qubits))

    # Step 7: Apply Hadamard gates again
    qc.h(range(num_qubits))

    return qc

# Modified function to allow specifying the number of Grover iterations
# Define Grover's algorithm with CUDA-enabled Grover iteration
def grovers_algorithm_gpu(num_qubits, target_state, iterations=65536):
    print("Setting up Grover's algorithm with CUDA-enabled diffusion steps...")
    circuit = QuantumCircuit(num_qubits, num_qubits)

    # Initialize qubits in superposition
    circuit.h(range(num_qubits))
    print("Initialized qubits in superposition.")

    # Apply QFT (if required)
    apply_qft(circuit, num_qubits)

    for iteration in range(iterations):
        print(f"Applying Grover iteration {iteration + 1} using CUDA...")

        # Apply the oracle with GPU acceleration if possible
        oracle_circuit = create_oracle_with_ancilla(num_qubits, target_state)
        oracle_matrix_gpu = cp.asarray(oracle_circuit.to_matrix())  # Oracle CUDA-ready matrix

        # Apply the oracle and GPU-enabled diffusion operator to the circuit
        circuit.compose(oracle_circuit, inplace=True)

        # Apply the Grover diffusion operator with CUDA acceleration
        diffusion_operator = create_diffusion_operator_gpu(num_qubits)
        circuit.compose(diffusion_operator, inplace=True)
        print("Applied oracle and diffusion operator using CUDA.")

    # Measure the qubits
    circuit.measure(range(num_qubits), range(num_qubits))
    print("Measurement operation added to circuit.")
    
    return circuit

# Quantum Brute-Force search using Grover's algorithm
def grovers_bruteforce(target_address, keyspace_size=None, num_qubits=125, iterations=65536):
    if keyspace_size is None:
        keyspace_size = 0x1fffffffffffffffffffffffffffffff - 0x10000000000000000000000000000000 + 1

    # Choose a random target state within the keyspace
    target_state = random.randint(0, keyspace_size - 1)
    print(f"Chosen target state: {target_state}")

    # Calculate iterations if not provided
    if iterations is None:
        iterations = 65536

    # Initialize Grover's algorithm circuit
    circuit = grovers_algorithm_gpu(num_qubits, target_state, iterations)

    circuit.measure_all()
    return circuit

def retrieve_job_result(job_id, target_address):
    print(f"Retrieving job result for job ID: {job_id}...")
    quantum_registers = 125  # Use 125 qubits for the search
    try:
        job = service.job(job_id)
        result = job.result()

        # Retrieve measurement results
        counts = result.get_counts()
        print(f"Measurement counts retrieved: {counts}")

        # Sort the counts by frequency
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        # Check for the target address
        for bin_key, count in sorted_counts:
            # Check if the key matches the expected size
            if len(bin_key) < quantum_registers:
                bin_key = bin_key.zfill(quantum_registers)
            elif len(bin_key) > quantum_registers:
                bin_key = bin_key[:quantum_registers]

            # Convert to compressed Bitcoin address
            private_key_hex = binary_to_hex(bin_key)
            compressed_address = private_key_to_compressed_address(private_key_hex)

            if compressed_address == target_address:
                print(f"Private key found: {private_key_hex}")
                with open('boom.txt', 'a') as file:
                    file.write(f"Private key: {private_key_hex}\nCompressed Address: {compressed_address}\n\n")
                return private_key_hex, compressed_address

        print("No matching private key found.")
        return None, None
    except Exception as e:
        print(f"Error retrieving job result: {e}")
        return None, None

# Function to convert binary to hex
def binary_to_hex(bin_key):
    bin_key = bin_key.zfill(128)  # Ensure 128-bit padding
    return hex(int(bin_key, 2))[2:].zfill(64)

def retrieve_job_result(job_id, target_address, quantum_registers):
    """Retrieve job results and check for valid private keys."""
    print(f"Retrieving job result for job ID: {job_id}...")
    service = QiskitRuntimeService()

    try:
        # Retrieve job result from the quantum device
        job = service.job(job_id)
        job_result = job.result()
        print(f"Job result retrieved for job ID {job_id}")
    except Exception as e:
        print(f"Error retrieving job result: {e}")
        return None, None

    try:
        # Access the measurement results (which are binary strings like '010101')
        counts = job_result.get_counts()
        print("Counts retrieved from job:", counts)

        # Check each binary result for a valid private key
        for bin_key, count in counts.items():
            bin_key = bin_key.strip()

            # Ensure the key is exactly 125 bits
            if len(bin_key) < quantum_registers:
                bin_key = bin_key.ljust(quantum_registers, '0')
            elif len(bin_key) > quantum_registers:
                bin_key = bin_key[:quantum_registers]

            print(f"\nChecking binary key (first 125 bits): {bin_key} with length {len(bin_key)}")
            print(f"Key count: {count} times generated")

            # Convert binary string to hex
            private_key_hex = binary_to_hex(bin_key)
            if private_key_hex is None:
                continue  # Skip if conversion failed

            # Convert to compressed Bitcoin address
            compressed_address = private_key_to_compressed_address(private_key_hex)

            # Check if the private key produces the target Bitcoin address
            if compressed_address == target_address:
                print(f"Valid private key found: {private_key_hex}")

                # Save the valid private key and address to boom.txt
                with open('boom.txt', 'a') as file:
                    file.write(f"Private Key: {private_key_hex}\n")
                    file.write(f"Compressed Address: {compressed_address}\n\n")

                return private_key_hex, compressed_address

        print("No matching private key found.")
        return None, None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

def quantum_brute_force(public_key_x: int, g_x: int, g_y: int, p: int, min_range: int, max_range: int) -> int:
    """Main function to perform quantum brute-force search for private keys using Public-Key-Hash."""
    if max_range <= min_range:
        raise ValueError("max_range must be greater than min_range.")

    target_address = '1PXAyUB8ZoH3WD8n5zoAthYjN15yN5CVq5'
    quantum_registers = 125  # Use 125 qubits for the search
    private_key = None
    attempt = 0
    failed_backends = set()
    num_ancillas = 1
    num_iterations = 65536

    service = QiskitRuntimeService()

    while private_key is None:
        attempt += 1
        print(f"Attempt {attempt}...")

        # Initialize quantum circuit with qubits and ancillas
        circuit = QuantumCircuit(quantum_registers + num_ancillas, quantum_registers)
        ancilla_register = QuantumRegister(num_ancillas, name='ancilla')
        circuit.add_register(ancilla_register)
        print("Quantum circuit initialized.")

        # Apply Hadamard gates to all qubits to create superposition
        circuit.h(range(quantum_registers))
        print("Hadamard gates applied.")

        # Apply Grover's iterations
        print(f"Applying Grover's iterations ({num_iterations} iterations).")
        for _ in range(num_iterations):
            # Oracle to mark correct private keys
            grover_oracle(circuit, range(quantum_registers), public_key_x, g_x, g_y, p, ancilla_register[0])
            # Apply the diffusion operator here (if you have defined it)

        # Measure the qubits after the iterations
        circuit.measure(range(quantum_registers), range(quantum_registers))
        print("Measurement operation added to circuit.")

        # Get a list of available backends
        available_backends = service.backends() 
        backend = service.backend('ibm_sherbrooke')      
        print(f"Selected backend: {backend}")        

        # Transpile and run the circuit with optimization level 3
        print("Transpiling the circuit for the selected backend.")
        transpiled_circuit = transpile(circuit, backend=backend, optimization_level=3)
        print("Circuit transpiled.")

        job = backend.run([transpiled_circuit], shots=8192)
        job_id = job.job_id()
        print(f"Job ID: {job_id}")


        # Retrieve the job result and check for valid private key
        found_key, compressed_address = retrieve_job_result(job_id, target_address, quantum_registers)

        if found_key:
            print(f"Found matching private key: {found_key}")
            return found_key  # Return the valid private key if found
        else:
            print("No matching key found.")
            break  # Break the loop if no key found

    return None

def main():
    target_address = '1PXAyUB8ZoH3WD8n5zoAthYjN15yN5CVq5'
    public_key_x_hex = "0233709eb11e0d4439a729f21c2c443dedb727528229713f0065721ba8fa46f00e"
    public_key_x = int(public_key_x_hex[2:], 16)
    num_qubits = 125
    iterations = 65536  # The number of iterations you want
    # Elliptic curve parameters
    g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199B0C75643B8F8E4F
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    min_range = 0x10000000000000000000000000000000  # Adjust this range as needed
    max_range = 0x1fffffffffffffffffffffffffffffff  # Adjust to fit within 125 bits

    while True:  # Loop indefinitely until the private key is found
        print("Starting new quantum brute-force search...")

        private_key = quantum_brute_force(public_key_x, g_x, g_y, p, min_range, max_range)

        if private_key is not None:
            # Format the private key as a hex string with padded zeros
            private_key_hex = f"{private_key:064x}"  # Format to 64 characters with leading zeros
            print(f"Found private key: {private_key_hex}")
            found_address = private_key_to_compressed_address(private_key_hex)
            print(f"Corresponding Bitcoin address: {found_address}")

            # Save the found private key to boomqft.txt
            with open("boomqft.txt", "w") as f:
                f.write(f"Found private key: {private_key_hex}\n")
                f.write(f"Corresponding Bitcoin address: {found_address}\n")
            print("Private key and corresponding address saved to boomqft.txt.")
            break  # Exit the loop once the key is found
        else:
            print("Private key not found in the specified range. Retrying...")

if __name__ == "__main__":
    main()
    
