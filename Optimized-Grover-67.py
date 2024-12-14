#i Realy hope you get me some Donation for the Quantum project_ 1NEJcwfcEm7Aax8oJNjRUnY3hEavCjNrai /////
#======================================================================================================
#import qctrl
#import qctrlvisualizer as qv
#import fireopal as fo
#import boulderopal as bo
#from qctrl import QAQA, QPE, QEC
from qiskit.circuit.controlflow.break_loop import BreakLoopPlaceholder
from qiskit.circuit.library import ZGate, MCXGate
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit.primitives import SamplerResult
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from collections import Counter
from Crypto.Hash import RIPEMD160, SHA256  # Import from pycryptodome
from ecdsa import SigningKey, SECP256k1
from bitarray import bitarray
from qiskit_aer.primitives import SamplerV2  # for simulator
from qiskit_ibm_runtime import SamplerV2 as real_sampler  # for hardware
from qiskit_aer import AerSimulator, Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService , Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister,  AncillaRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit.quantum_info import Operator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector
from qiskit_algorithms import Grover, AmplificationProblem
#from qiskit_algorithms.optimizers import AmplificationProblem, CustomCircuitOracle
import numpy as np
import hashlib
import random
import base58
import math
from math import ceil, log2
import time

#from qctrlcommons.preconditions import check_argument_integer
# Setup Fire-Opal
#bo.cloud.set_organization("aimen-eldjoundi-gmail")
#bo.authenticate_qctrl_account("ZGmK5goCNAKxiOfFCMJZbShiX7jk8llgKGBVASGSerYNYE131L")
#bo.cloud.request_machines(1)
#fo.config.configure_organization(organization_slug="aimen-eldjoundi-gmail")
#fo.authenticate_qctrl_account("ZGmK5goCNAKxiOfFCMJZbShiX7jk8llgKGBVASGSerYNYE131L")

# Load IBMQ account using QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel='ibm_quantum',
    token='_ADD_YOUR_IBM_QUANTUM_TOKEN_',  # Replace with your actual token
    instance='ibm-q/open/main',
    overwrite=True,
    set_as_default=True
)

# Load the "open" credentials
service = QiskitRuntimeService()

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

def apply_qft(grover_circuit, num_qubits):
    """
    Applies the Quantum Fourier Transform (QFT) on the first `num_qubits` qubits of the circuit.
    num_qubits = 67
    Args:
        circuit (QuantumCircuit): The quantum circuit to which QFT is applied.
        num_qubits (int): The number of qubits to include in the QFT.
    """
    # Append the QFT library object for validation
    qft = QFT(num_qubits)
    grover_circuit.append(qft, range(num_qubits))
    
    # Manual implementation of QFT for additional customization
    for j in range(num_qubits):
        grover_circuit.h(j)
        for k in range(j + 1, num_qubits):
            grover_circuit.cp(np.pi / (2 ** (k - j)), j, k)
    
    # Swap operations to reverse the order of qubits
    for j in range(num_qubits // 2):
        grover_circuit.swap(j, num_qubits - j - 1)


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

# ECC Functions for Bitcoin Address Generation
def mod_inverse(a, p):
    """Returns modular inverse of a under modulo p."""
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % p, p
    while low > 1:
        ratio = high // low
        nm, new_low = hm - lm * ratio, high - low * ratio
        lm, low, hm, high = nm, new_low, lm, low
    return lm % p

def point_addition(x1, y1, x2, y2, p):
    """Performs point addition on the elliptic curve."""
    if x1 == x2 and y1 == y2:
        return point_doubling(x1, y1, p)
    lam = ((y2 - y1) * mod_inverse(x2 - x1, p)) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return x3, y3

def point_doubling(x1, y1, p):
    """Performs point doubling on the elliptic curve."""
    lam = ((3 * x1 * x1) * mod_inverse(2 * y1, p)) % p
    x3 = (lam * lam - 2 * x1) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return x3, y3

def scalar_multiplication(k, x, y, p):
    """Performs scalar multiplication of a point on the elliptic curve."""
    x_res, y_res = x, y
    k_bin = bin(k)[2:]
    for bit in k_bin[1:]:
        x_res, y_res = point_doubling(x_res, y_res, p)
        if bit == '1':
            x_res, y_res = point_addition(x_res, y_res, x, y, p)
    return x_res, y_res

def public_key_to_btc_address(x, y, compressed=True):
    """Generates a Bitcoin address from the public key."""
    if compressed:
        prefix = b'\x02' if y % 2 == 0 else b'\x03'
        public_key = prefix + x.to_bytes(32, 'big')
    else:
        public_key = b'\x04' + x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
    sha256 = hashlib.sha256(public_key).digest()
    ripemd160 = hashlib.new('ripemd160', sha256).digest()
    return ripemd160


def retrieve_job_result(job_id, target_address):
    print(f"Retrieving job result for job ID: {job_id}...")
    num_qubits = 67  # Use 67 qubits for the search
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
            if len(bin_key) < num_qubits:
                bin_key = bin_key.zfill(num_qubits)
            elif len(bin_key) > num_qubits:
                bin_key = bin_key[:num_qubits]

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

def retrieve_job_result(job_id, target_address, num_qubits):
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

            # Ensure the key is exactly 67 bits
            if len(bin_key) < num_qubits:
                bin_key = bin_key.ljust(num_qubits, '0')
            elif len(bin_key) > num_qubits:
                bin_key = bin_key[:num_qubits]

            print(f"\nChecking binary key (first 67 bits): {bin_key} with length {len(bin_key)}")
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


def grover_oracle(grover_circuit, target_private_key, num_qubits, public_key_hash, g_x, g_y, p, ancilla):
    """Oracle for Grover's algorithm that checks if the current private key qubits 
    match the target private key via scalar multiplication to compare public key hashes."""
    keyspace_size = 0x7ffffffffffffffff - 0x40000000000000000 + 1
    grover_circuit.barrier()
    num_qubits = 67  # Ensure this matches the actual number of qubits used in the circuit
    
    # Randomly generate a private key (target_state) within the keyspace:
    target_private_key = random.randint(0, keyspace_size - 1)
    target_private_key_bin = format(target_private_key, f'0{num_qubits}b')  # Convert to binary string
    
    # Use scalar multiplication to compute the public key corresponding to the target private key
    computed_x, _ = scalar_multiplication(target_private_key, g_x, g_y, p)
    
    # Convert computed public key to hexadecimal
    computed_public_key_hex = format(computed_x, 'x').zfill(64)
    
    # Convert computed public key to Public-Key-Hash
    computed_public_key_hash = public_key_to_public_key_hash(computed_public_key_hex)

    # Create the oracle with the ancillary qubit (mark the target state)
    grover_circuit.x(num_qubits)  # Set ancilla to |1⟩

    # Flip the bits to create the target state
    for i, bit in enumerate(bin(target_private_key)[2:].zfill(num_qubits)):
        if bit == '0':
            grover_circuit.x(i)

    # Apply a controlled Z (CZ) gate to mark the target state
    grover_circuit.cz(num_qubits, num_qubits - 1)  # Control on ancilla, target on last qubit

    # Flip back the bits to restore original state
    for i, bit in enumerate(bin(target_private_key)[2:].zfill(num_qubits)):
        if bit == '0':
            grover_circuit.x(i)

    # If the computed public key hash matches the provided public key hash, mark the state
    if computed_public_key_hash == public_key_hash:
        # Flip qubits to match the target private key state
        for i, bit in enumerate(target_private_key_bin):
            if bit == '0':
                grover_circuit.x(i)

        # Apply the ancilla qubit to mark the state (controlled NOT on all qubits)
        grover_circuit.mcx(range(num_qubits), ancilla)

        # Flip the qubits back to their original state after the mark
        for i, bit in enumerate(target_private_key_bin):
            if bit == '0':
                grover_circuit.x(i)

    # Flip the ancilla qubit if the computed public key hash matches the provided public key hash
    if computed_public_key_hash == public_key_hash:
        grover_circuit.x(ancilla)

    grover_circuit.barrier()

  
def quantum_brute_force(public_key_hash: str, g_x: int, g_y: int, p: int, min_range: int, max_range: int, target_private_key: int, iterations: int = None) -> int:
    """Perform quantum brute-force search using Grover's algorithm."""
    if max_range <= min_range:
        raise ValueError("max_range must be greater than min_range.")

    # Calculate iterations if not provided
    if iterations is None:
        iterations = int(np.sqrt(float(max_range - min_range)))  # Default iterations as sqrt(N)

    num_qubits = 67
    num_ancillas = 1
    quantum_registers = 67  # Use 67 qubits for the search
    private_key = None
    attempt = 0
    failed_backends = set()
    print(f"Starting quantum brute-force search with {num_qubits} total qubits...")

    while private_key is None:
        attempt += 1
        print(f"Attempt {attempt}...")

        # Initialize quantum circuit with qubits and ancillas
        grover_circuit = QuantumCircuit(num_qubits, num_qubits)
        ancilla_register = QuantumRegister(num_ancillas, name='ancilla')
        grover_circuit.add_register(ancilla_register)
        print("Quantum circuit initialized.")

        # Apply Hadamard gates to all qubits to create superposition
        grover_circuit.h(range(quantum_registers))
        print("Initialized qubits in Superposition.")
        print("Hadamard gates applied.")
        
        # Use the passed target_private_key for oracle creation
        target_private_key_bin = format(target_private_key, f'0{num_qubits}b')  # Binary string of target private key
        print(f"Target Private Key (in binary): {target_private_key_bin}")

        # Create the oracle with the ancillary qubit
        grover_oracle(grover_circuit, target_private_key, num_qubits, public_key_hash, g_x, g_y, p, ancilla_register[0])

        # Step 3: Apply Quantum Fourier Transform (QFT)
        apply_qft(grover_circuit, num_qubits)

        # Perform Grover's iterations inside the circuit
        grover = Grover(iterations=iterations)

        # Inverse QFT for result extraction
        grover_circuit.append(QFT(num_qubits).inverse(), range(num_qubits))

        # Add measurements
        grover_circuit.measure(range(num_qubits), range(num_qubits))
        print("Measurement operation added to circuit.")

        # Step 7: Select backend and execute
        available_backends = service.backends()
        backend = service.backend('ibm_brisbane')  # Choose a backend
        print(f"Selected backend: {backend}")

        # Transpile and execute on the backend
        print("Transpiling the circuit for the selected backend.")
        transpiled_circuit = transpile(grover_circuit, backend=backend, optimization_level=3)
        print(f"Circuit depth: {transpiled_circuit.depth()}")

        job = backend.run([transpiled_circuit], shots=8192)
        job_id = job.job_id()
        print(f"Job ID: {job_id}")

        # Retrieve the job result and check for valid private key
        private_key, compressed_address = retrieve_job_result(job_id, public_key_hash, num_qubits)

        if private_key:
            print(f"Found matching private key: {private_key}")
            return private_key  # Return the valid private key if found
        else:
            print("No matching key found.")
            break  # Break the loop if no key found

    return None


"""Main function to perform quantum brute-force search for private keys using Public-Key-Hash."""
def main():
    target_address = '1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9'  # Replace with the Target Bitcoin address
    public_key_hash = "739437bb3dd6d1983e66629c5f08c70e52769371"  # Add here Pubhash ripemd160
    num_qubits = 67     # Adjust this as needed (current for puzzle 67)   

    g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199B0C75643B8F8E4F
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

    min_range = 0x40000000000000000
    max_range = 0x7ffffffffffffffff

    iterations = int(np.sqrt(float(max_range - min_range)))  # Corrected calculation
    print(f"Calculated iterations: {iterations}")

    while True:
        print("Starting a new quantum brute-force search...")
        private_key = quantum_brute_force(public_key_hash, g_x, g_y, p, min_range, max_range, iterations)

        if private_key is not None:
            private_key_hex = f"{private_key:064x}"
            print(f"Found private key: {private_key_hex}")
            found_address = private_key_to_compressed_address(private_key_hex)
            print(f"Corresponding Bitcoin address: {found_address}")

            # Save results to file
            with open("boomqft.txt", "w") as f:
                f.write(f"Found private key: {private_key_hex}\n")
                f.write(f"Corresponding Bitcoin address: {found_address}\n")
            print("Private key and corresponding address saved to boomqft.txt.")
            break
        else:
            print("Private key not found. Retrying...")

if __name__ == "__main__":
    main()
