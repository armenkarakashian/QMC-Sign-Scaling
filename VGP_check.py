"""
3x3 lattice Hamiltonian (nearest neighbors):
H = sum_{<i,j>} ( h0^(i,j) (Z_i Z_j - I) + i * h1^(i,j) (Z_i + Z_j) ) (X_i X_j)

h0^(i,j) > 0 and h1^(i,j) ∈ R are chosen at random (all h1^(i,j) share the same sign).
"""

from __future__ import annotations
import numpy as np
import sys

def adjacency_pairs_3x3():
    pairs = []
    # index mapping: (r, c) -> r*3 + c  with r,c in {0,1,2}
    for r in range(3):
        for c in range(3):
            i = r*3 + c
            if c < 2: 
                j = r*3 + (c+1)
                pairs.append((i, j))
            if r < 2: 
                j = (r+1)*3 + c
                pairs.append((i, j))
    return pairs

def single_op(n_qubits: int, pos: int, op: np.ndarray) -> np.ndarray:
    I2 = np.eye(2, dtype=complex)
    mats = [I2]*n_qubits
    mats = [I2 if k != pos else op.astype(complex) for k in range(n_qubits)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def precompute_single_qubit_ops(n_qubits: int):
    X = np.array([[0, 1],[1, 0]], dtype=complex)
    Z = np.array([[1, 0],[0,-1]], dtype=complex)
    X_ops = {}
    Z_ops = {}
    for q in range(n_qubits):
        X_ops[q] = single_op(n_qubits, q, X)
        Z_ops[q] = single_op(n_qubits, q, Z)
    I_full = np.eye(2**n_qubits, dtype=complex)
    return X_ops, Z_ops, I_full

def build_hamiltonian(n_qubits: int, seed: int, h0_range=(0.1, 0.5), h1_range=(-0.5, 0.5)):
    rng = np.random.default_rng()
    pairs = adjacency_pairs_3x3()  # 12 edges
    X_ops, Z_ops, I_full = precompute_single_qubit_ops(n_qubits)

    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    h0_vals = {}
    h1_vals = {}

    sign = 1
    if np.random.randint(0,2) == 0:
        sign = -1

    for (i, j) in pairs:
        h0 = rng.uniform(*h0_range)
        h1 = sign*rng.uniform(*h0_range)

        Zi = Z_ops[i]
        Zj = Z_ops[j]
        Xi = X_ops[i]
        Xj = X_ops[j]

        ZiZj = Zi @ Zj
        XiXj = Xi @ Xj

        term = (h0 * (ZiZj - I_full) + 1j * h1 * (Zi + Zj)) @ XiXj
        H += term

    return H, h0_vals, h1_vals

def trace_expm_via_eigs(A: np.ndarray) -> complex:
    evals = np.linalg.eigvals(A)
    return np.sum(np.exp(evals))

def main(seed: int):
    n = 9
    H, h0_vals, h1_vals = build_hamiltonian(n, seed=seed)

    H_prime = np.abs(H - np.diag(np.diag(H))) - np.diag(np.diag(H))

    tr_exp_Hprime = trace_expm_via_eigs(H_prime)
    tr_exp_minusH = trace_expm_via_eigs(-H)
    diff = tr_exp_Hprime - tr_exp_minusH
    return diff

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} number_of_trials")
        sys.exit(1)

    M = int(sys.argv[1])
    tol = 1e-9 

    for trial in range(M):
        diff = main(seed=trial)
        if abs(diff) > tol:
            raise RuntimeError(
                f"Trial {trial}: difference != 0 (diff = {diff})"
            )

    print(f"All Hamiltonians are VGP across {M} trials.")