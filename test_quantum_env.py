# Test 1: PennyLane works
import pennylane as qml

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

print("Test 1 - PennyLane:", circuit())

# Test 2: Qiskit Aer works
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sim = AerSimulator()
job = sim.run(qc, shots=1000)
result = job.result()
print("Test 2 - Qiskit Aer:", result.get_counts())

# Test 3: PennyLane can use Aer as backend
dev_aer = qml.device("qiskit.aer", wires=2)

@qml.qnode(dev_aer)
def circuit_aer():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

print("Test 3 - PennyLane + Aer bridge:", circuit_aer())