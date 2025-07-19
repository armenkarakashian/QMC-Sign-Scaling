import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import gc, sys, psutil, os

def mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

pauli = {'X': X, 'Y': Y, 'Z': Z}

hamiltonian_data = """
1.0 1 X 2 X
1.0 1 Y 2 Y
1.0 1 Z 2 Z
1.0 1 X 3 X
1.0 1 Y 3 Y
1.0 1 Z 3 Z
-1.0 2 X 3 X
-1.0 2 Y 3 Y
-1.0 2 Z 3 Z
1.0 2 X 4 X
1.0 2 Y 4 Y
1.0 2 Z 4 Z
1.0 3 X 4 X
1.0 3 Y 4 Y
1.0 3 Z 4 Z
1.0 3 X 5 X
1.0 3 Y 5 Y
1.0 3 Z 5 Z
-1.0 4 X 5 X
-1.0 4 Y 5 Y
-1.0 4 Z 5 Z
1.0 4 X 6 X
1.0 4 Y 6 Y
1.0 4 Z 6 Z
1.0 5 X 6 X
1.0 5 Y 6 Y
1.0 5 Z 6 Z
1.0 5 X 7 X
1.0 5 Y 7 Y
1.0 5 Z 7 Z
-1.0 6 X 7 X
-1.0 6 Y 7 Y
-1.0 6 Z 7 Z
1.0 6 X 8 X
1.0 6 Y 8 Y
1.0 6 Z 8 Z
1.0 7 X 8 X
1.0 7 Y 8 Y
1.0 7 Z 8 Z
1.0 7 X 9 X
1.0 7 Y 9 Y
1.0 7 Z 9 Z
-1.0 8 X 9 X
-1.0 8 Y 9 Y
-1.0 8 Z 9 Z
1.0 8 X 10 X
1.0 8 Y 10 Y
1.0 8 Z 10 Z
1.0 9 X 10 X
1.0 9 Y 10 Y
1.0 9 Z 10 Z
1.0 9 X 11 X
1.0 9 Y 11 Y
1.0 9 Z 11 Z
-1.0 10 X 11 X
-1.0 10 Y 11 Y
-1.0 10 Z 11 Z
1.0 10 X 12 X
1.0 10 Y 12 Y
1.0 10 Z 12 Z
1.0 11 X 12 X
1.0 11 Y 12 Y
1.0 11 Z 12 Z
"""

terms = []
for line in hamiltonian_data.strip().splitlines():
    c, q1, a1, q2, a2 = line.strip().split()
    terms.append((float(c), int(q1), a1, int(q2), a2))

n_qubits = 12
dim = 2 ** n_qubits

H = np.zeros((dim, dim), dtype=np.complex128)

def build_two_body_op(q1, a1, q2, a2):
    ops = []
    for idx in range(1, n_qubits + 1):
        if idx == q1:
            ops.append(pauli[a1])
        elif idx == q2:
            ops.append(pauli[a2])
        else:
            ops.append(I)
    return reduce(np.kron, ops)

for coeff, q1, a1, q2, a2 in terms:
    term_op = build_two_body_op(q1, a1, q2, a2)
    H += -coeff * term_op  
    del term_op
    gc.collect()

eig_H = np.linalg.eigvalsh(H)

diag_H = np.diag(H).real.copy()
H[:] = np.abs(H)  
H[np.diag_indices(dim)] = -diag_H
del diag_H
gc.collect()

eig_Habs = np.linalg.eigvalsh(H.real)  

betas = np.array([0.5, 1, 1.5, 2, 2.5, 3, 5, 10], dtype=np.float64)
sgn_vals = []

for beta in betas:
    Z_H = np.exp(-beta * eig_H).sum()
    Z_Habs = np.exp(beta * eig_Habs).sum()
    sgn_vals.append(Z_H / Z_Habs)

sgn_vals = np.array(sgn_vals)

plt.figure(figsize=(6, 4))
plt.semilogy(betas, sgn_vals, marker='^', color='blue', linestyle='-', label=r'$\langle \mathrm{sgn}\rangle$')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\langle \mathrm{sgn}\rangle$')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()