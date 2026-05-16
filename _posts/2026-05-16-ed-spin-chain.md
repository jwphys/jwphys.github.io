---
layout: post
title: "Exact Diagonalization of 1D Spin Chains and Ladder Models in Python"
date: 2026-05-16
tags: [python, exact-diagonalization, spin-chain, quantum-magnetism]
lang: en
---

Exact diagonalization (ED) is a powerful numerical method for studying finite quantum many-body systems. By constructing and diagonalizing the Hamiltonian matrix in a chosen basis, we can obtain exact ground-state energies, excited states, and correlation functions without approximation (up to numerical precision).

## 1. Spin-1/2 Heisenberg Chain

The Hamiltonian for the 1D spin-1/2 Heisenberg chain is:

$$H = J \sum_{i=1}^{L-1} \mathbf{S}_i \cdot \mathbf{S}_{i+1}$$

We work in the basis of $S^z$ eigenstates: $|\sigma_1, \sigma_2, \dots, \sigma_L\rangle$, where $\sigma_i = \uparrow$ or $\downarrow$.

### Python Implementation

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def pauli_matrices():
    """Pauli matrices"""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return sx, sy, sz

def spin_operators(S=0.5):
    """Spin operators for given spin quantum number"""
    sx, sy, sz = pauli_matrices()
    return S * sx, S * sy, S * sz

def heisenberg_chain(L, J=1.0):
    """Build Heisenberg chain Hamiltonian in the full Hilbert space"""
    dim = 2 ** L
    sx, sy, sz = spin_operators(0.5)
    eye2 = np.eye(2, dtype=complex)

    H = np.zeros((dim, dim), dtype=complex)

    for i in range(L - 1):
        # Build S_i · S_{i+1} as a Kronecker product
        op_list_before = [eye2] * i
        op_list_after = [eye2] * (L - i - 2)

        # Sx_i Sx_{i+1}
        op_x = op_list_before + [sx, sx] + op_list_after
        # Sy_i Sy_{i+1}
        op_y = op_list_before + [sy, sy] + op_list_after
        # Sz_i Sz_{i+1}
        op_z = op_list_before + [sz, sz] + op_list_after

        def kron_product(ops):
            result = ops[0]
            for op in ops[1:]:
                result = np.kron(result, op)
            return result

        H += J * (kron_product(op_x) + kron_product(op_y) + kron_product(op_z))

    return H

# Example: L=8 chain, compute ground state
L = 8
H = heisenberg_chain(L)
eigvals, eigvecs = np.linalg.eigh(H)
print(f"Ground state energy (L={L}): {eigvals[0]:.6f}")
print(f"First excited state: {eigvals[1]:.6f}")
```

## 2. Spin Ladder Model

A two-leg spin ladder is described by:

$$H = J_\parallel \sum_{i, \alpha} \mathbf{S}_{i,\alpha} \cdot \mathbf{S}_{i+1,\alpha} + J_\perp \sum_i \mathbf{S}_{i,1} \cdot \mathbf{S}_{i,2}$$

where $\alpha = 1, 2$ labels the two legs, $J_\parallel$ is the intra-leg coupling, and $J_\perp$ is the rung coupling.

```python
def spin_ladder(L, J_parallel=1.0, J_perp=1.0):
    """Build spin ladder Hamiltonian for L rungs (2L spins total)"""
    dim = 2 ** (2 * L)
    sx, sy, sz = spin_operators(0.5)
    eye2 = np.eye(2, dtype=complex)

    H = np.zeros((dim, dim), dtype=complex)

    def kron_n(ops, total):
        """Kronecker product of ops list, padded to total length with identities"""
        padded = []
        op_idx = 0
        for pos in range(total):
            if op_idx < len(ops) and ops[op_idx][0] == pos:
                padded.append(ops[op_idx][1])
                op_idx += 1
            else:
                padded.append(eye2)
        result = padded[0]
        for m in padded[1:]:
            result = np.kron(result, m)
        return result

    # Intra-leg couplings (leg 1: sites 0..L-1, leg 2: sites L..2L-1)
    for leg in range(2):
        offset = leg * L
        for i in range(L - 1):
            ops_x = [(offset + i, sx), (offset + i + 1, sx)]
            ops_y = [(offset + i, sy), (offset + i + 1, sy)]
            ops_z = [(offset + i, sz), (offset + i + 1, sz)]
            H += J_parallel * (
                kron_n(ops_x, 2*L) + kron_n(ops_y, 2*L) + kron_n(ops_z, 2*L)
            )

    # Rung couplings
    for i in range(L):
        ops_x = [(i, sx), (i + L, sx)]
        ops_y = [(i, sy), (i + L, sy)]
        ops_z = [(i, sz), (i + L, sz)]
        H += J_perp * (
            kron_n(ops_x, 2*L) + kron_n(ops_y, 2*L) + kron_n(ops_z, 2*L)
        )

    return H

# Example: 4 rungs (8 spins total)
L_rung = 4
H_ladder = spin_ladder(L_rung, J_parallel=1.0, J_perp=0.5)
eigvals = np.linalg.eigvalsh(H_ladder)
print(f"Ladder ground state (L={L_rung}): {eigvals[0]:.6f}")
print(f"Spin gap: {eigvals[1] - eigvals[0]:.6f}")
```

## 3. Measuring Observables

Once we have the ground-state wavefunction $|\psi_0\rangle$, we can compute various observables:

```python
def expectation_value(state, operator):
    return state.conj().T @ operator @ state

def sz_correlation(state, L, i, j):
    """Compute <S^z_i S^z_j>"""
    sx, sy, sz = spin_operators(0.5)
    eye2 = np.eye(2, dtype=complex)

    ops_before_i = [eye2] * i
    ops_between = [eye2] * (j - i - 1)
    ops_after_j = [eye2] * (L - j - 1)

    op = ops_before_i + [sz] + ops_between + [sz] + ops_after_j
    result = op[0]
    for m in op[1:]:
        result = np.kron(result, m)
    return expectation_value(state, result).real
```

## Summary

Exact diagonalization provides exact results for small systems (typically up to $L \sim 16$ for spin-1/2 chains, or $L \sim 12$ for ladders). It serves as a benchmark for approximate methods like DMRG and allows detailed study of entanglement spectra, correlation functions, and finite-size scaling.
