import tensorflow as tf
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
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Operator
import math

# Load IBMQ account using QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel='ibm_quantum',
    token='_IBM_QUANTUM_TOKEN_',  # Replace with your actual token
    instance='ibm-q/open/main',
    overwrite=True,
    set_as_default=True
)

# Load the service
service = QiskitRuntimeService()

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# Constants for elliptic curve operations (example curve secp256k1 values)
p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
a = 0
g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

def scalar_multiplication(k, x, y, p):
    """Perform scalar multiplication using the double-and-add method."""
    k_bin = tf.convert_to_tensor([int(bit) for bit in bin(k)[2:]], dtype=tf.int32)
    current_x, current_y = x, y
    result_x, result_y = None, None

    for bit in k_bin:
        if result_x is not None:
            # Point doubling
            slope = (3 * current_x**2 + a) * pow(2 * current_y, -1, p)
            slope = slope % p
            new_x = (slope**2 - 2 * current_x) % p
            new_y = (slope * (current_x - new_x) - current_y) % p
            current_x, current_y = new_x, new_y
        
        if bit == 1:
            if result_x is None:
                result_x, result_y = current_x, current_y
            else:
                # Point addition
                slope = (current_y - result_y) * pow(current_x - result_x, -1, p)
                slope = slope % p
                new_x = (slope**2 - result_x - current_x) % p
                new_y = (slope * (result_x - new_x) - result_y) % p
                result_x, result_y = new_x, new_y

    return result_x, result_y

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

def create_oracles_shors_tpu(num_qubits, public_key_hash, keyspace_start, keyspace_end):
    """Create the oracle matrix for Grover's algorithm using TensorFlow and TPU."""
    if keyspace_start > keyspace_end:
        raise ValueError("keyspace_start must be less than or equal to keyspace_end.")
    
    oracle_matrix = tf.eye(2**num_qubits, dtype=tf.complex128)

    # Generate keyspace range
    keyspace_range = tf.range(keyspace_start, keyspace_end + 1, dtype=tf.int64)

    for k in keyspace_range:
        computed_x, _ = scalar_multiplication(k.numpy(), g_x, g_y, p)
        computed_public_key_hex = hex(computed_x)[2:].zfill(64)
        computed_public_key_hash = public_key_to_public_key_hash(computed_public_key_hex)

        if computed_public_key_hash == public_key_hash:
            # Flip the oracle value (simulating Z-gate operation)
            oracle_matrix = tf.tensor_scatter_nd_update(
                oracle_matrix,
                indices=[[k - keyspace_start]],  # Adjust index relative to keyspace_start
                updates=[-1]
            )

    return oracle_matrix

def shors_oracle_with_keyspace_tpu(circuit, q_x, anc, public_key_hash, keyspace_start, keyspace_end):
    """Implement Shor's oracle with keyspace restriction using TensorFlow and TPU."""
    if keyspace_start > keyspace_end:
        raise ValueError("keyspace_start must be less than or equal to keyspace_end.")
    
    print(f"Applying Shor's oracle with TPU in keyspace [{keyspace_start}, {keyspace_end}]...")

    # Generate keyspace range
    keyspace_range = tf.range(keyspace_start, keyspace_end + 1, dtype=tf.int64)

    for k in keyspace_range:
        index = k - keyspace_start  # Adjust index to fit the current keyspace
        if index < len(q_x):
            computed_x, _ = scalar_multiplication(k.numpy(), g_x, g_y, p)
            computed_public_key_hex = hex(computed_x)[2:].zfill(64)
            computed_public_key_hash = public_key_to_public_key_hash(computed_public_key_hex)

            if computed_public_key_hash == public_key_hash:
                circuit.mcx(q_x, anc[0])  # Use MCXGate with ancilla as the target

    print("Oracle applied with TPU-accelerated keyspace.")
    return circuit


# Convert binary string to hexadecimal
def binary_to_hex(bin_str):
    hex_str = hex(int(bin_str, 2))[2:].upper()
    return hex_str.zfill(64)  # Pad hex string to ensure it's 64 characters long (32 bytes)

# Retrieve the job result and search for the target
def retrieve_job_result(job_id, target_address):
    print(f"Retrieving job result for job ID: {job_id}...")
    try:
        job = service.job(job_id)
        result = job.result()

        # Print the job result to inspect its structure
        counts = result.get_counts()
        print(f"Measurement counts retrieved: {counts}")

        # Count occurrences of each unique measurement result
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        # Check if any of the keys correspond to the target Bitcoin address
        for bin_key, count in sorted_counts:
            private_key_hex = binary_to_hex(bin_key)

            # Convert to compressed Bitcoin address
            compressed_address = private_key_to_compressed_address(private_key_hex)

            if compressed_address is not None and compressed_address == target_address:
                print(f"Private key found: {private_key_hex}")

                # Save the found private key to boom.txt
                with open('boom.txt', 'a') as file:
                    file.write(f"Private key: {private_key_hex}\nCompressed Address: {compressed_address}\n\n")

                return private_key_hex, compressed_address  # Return the found key

        print("No matching private key found.")
        return None, None
    except Exception as e:
        print(f"Error while retrieving job result: {e}")
        return None, None

# Convert binary string to hexadecimal
def binary_to_hex(bin_str):
    hex_str = hex(int(bin_str, 2))[2:].upper()
    return hex_str.zfill(64)  # Pad hex string to ensure it's 64 characters long (32 bytes)

# Function to plot the result histogram of counts
def plot_result_histogram(counts):
    plot_histogram(counts)
    labels, values = zip(*counts.items())  # Unpack the counts dictionary
    if isinstance(labels[0], str) and all(set(label).issubset({'0', '1'}) for label in labels):

        hex_labels = [binary_to_hex(label) for label in labels]
    else:
        hex_labels = labels

    plt.bar(range(len(values)), values, tick_label=hex_labels)
    plt.xlabel('Private Keys (Hex)' if isinstance(hex_labels[0], str) else 'Private Key Candidates')
    plt.ylabel('Counts')
    plt.title('Histogram of Measurement Counts')
    plt.tight_layout()
    plt.show()

# Function to get the top 10 most frequent counts
def get_top_10_frequent(counts):
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top_10_counts = sorted_counts[:10]
    print(f"Top 10 most frequent counts: {top_10_counts}")
    
    private_key_candidates = [int(bitstring, 2) for bitstring, _ in top_10_counts]
    return private_key_candidates

def run_on_tpu():
    """Run Grover's algorithm oracle preparation on a TPU."""
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # Detect TPU
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope():
        num_qubits = 26
        public_key_hash = "bfebb73562d4541b32a02ba664d140b5a574792f"
        keyspace_start = 0x2000000
        keyspace_end = 0x3ffffff

        # Create the quantum oracle matrix
        oracle_matrix = create_oracles_shors_tpu(
            num_qubits,
            public_key_hash,
            keyspace_start,
            keyspace_end
        )

        print("Oracle Matrix:", oracle_matrix)


# Main quantum algorithm for private key recovery
def quantum_private_key_recovery(public_key_hash, num_qubits, target_address, keyspace_start, keyspace_end):
    while True:
        q_x = QuantumRegister(num_qubits, 'x')
        c = ClassicalRegister(num_qubits + 1, 'c')
        anc = QuantumRegister(1, 'anc')
        circuit = QuantumCircuit(q_x, anc, c)

        circuit.h(q_x)
        print("Superposition created.")

        circuit = shors_oracle_with_keyspace_tpu(circuit, q_x, anc, public_key_hash, keyspace_start, keyspace_end)
        print("Oracle applied.")

        circuit.append(QFT(num_qubits), q_x)
        print("QFT applied.")

        circuit.measure(q_x, c[:num_qubits])
        circuit.measure(anc[0], c[num_qubits])
        print("Measurement operation added.")

        backend = service.backend('ibm_kyiv')
        transpiled_circuit = transpile(circuit, backend=backend, optimization_level=3)
        print("Circuit transpiled.")

        job = backend.run([transpiled_circuit], shots=8192)
        job_id = job.job_id()
        print(f"Job ID: {job_id}")

        result = job.result()
        counts = result.get_counts()
        print("Result Counts:", counts)

        private_key_candidates = get_top_10_frequent(counts)
        found_key, compressed_address = retrieve_job_result(job_id, target_address)

        if found_key:
            print(f"Found matching private key: {found_key}")
            with open("boomm.txt", "w") as file:
                file.write(found_key)
            print(f"Completed all attempts. Found key: {found_key}")
            break
        
        print("Key not found, resubmitting job...")
        time.sleep(5)

if __name__ == "__main__":
    public_key_hash = "bfebb73562d4541b32a02ba664d140b5a574792f"  # Replace with your public key hash
    target_address = "1JVnST957hGztonaWK6FougdtjxzHzRMMg"  # Replace with the target Bitcoin address
    num_qubits = 26  # Adjust the number of qubits as needed
    keyspace_start = 0x2000000    # Set start of keyspace
    keyspace_end = 0x3ffffff   # Set end of keyspace

    quantum_private_key_recovery(public_key_hash, num_qubits, target_address, keyspace_start, keyspace_end)
    run_on_tpu()
