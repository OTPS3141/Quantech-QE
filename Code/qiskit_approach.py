from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli
import numpy as np

M = 2   #Number of Localizations
N = 2*M #total number of operators (Two positions times two spin polarizations)

#Define the Pauli matrices
I = Pauli(np.array([[1,0],[0,1]]), np.array([[0,0],[0,0]]))
X = Pauli(np.array([[0,1],[1,0]]), np.array([[1,0],[0,-1]]))
Y = Pauli(np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]]))
Z = Pauli(np.array([[1,0],[0,-1]]), np.array([[0,0],[0,0]]))

#Define the four basis states
Tu = QuantumCircuit(2)
Tu.x(0)
Td = QuantumCircuit(2)
Td.x(1)
Bu = QuantumCircuit(2)
Bu.x(0)
Bu.x(1)
Bd = QuantumCircuit(2)
Bd.x(0)
Bd.x(1)


