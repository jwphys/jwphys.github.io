---
layout: post
title: "Block Diagonalization: Leveraging Symmetries in Exact Diagonalization"
date: 2026-05-15
tags: [python, exact-diagonalization, symmetries, quantum-magnetism]
lang: en
---

Exact diagonalization of quantum Hamiltonians is severely limited by the exponential growth of the Hilbert space. However, by exploiting symmetries of the Hamiltonian, we can block-diagonalize the matrix and solve much larger systems. Here we implement block diagonalization using total $S^z$ and total momentum $k$ symmetries.

## 1. Total $S^z$ Symmetry

The Heisenberg model conserves total magnetization $S^z_{\text{tot}} = \sum_i S^z_i$. This means the Hamiltonian commutes with $S^z_{\text{tot}}$, and we can work in sectors of fixed total $S^z$.

### Basis Construction

```python
import numpy as np
from itertools import combinations

def sz_sector_basis(L, sz_target):
    """
    Generate all basis states with a given total S^z.
    States are represented as integers (bitstrings) where 1 = up, 0 = down.
    Returns list of basis states and their dimension.
    """
    n_up = sz_target + L // 2  # number of up spins needed
    if n_up < 0 or n_up > L:
        return [], 0

    basis = []
    for up_positions in combinations(range(L), n_up):
        state = 0
        for pos in up_positions:
            state |= (1 << pos)
        basis.append(state)

    return basis, len(basis)

def build_hamiltonian_sz_sector(L, J=1.0, sz_target=0):
    """
    Build Heisenberg Hamiltonian restricted to a fixed S^z sector.
    """
    basis, dim = sz_sector_basis(L, sz_target)
    if dim == 0:
        return None, None

    H = np.zeros((dim, dim), dtype=float)
    # Mapping from bitstring to basis index
    state_to_idx = {state: i for i, state in enumerate(basis)}

    # Precompute bond operator contributions
    for idx, state in enumerate(basis):
        for i in range(L - 1):
            si = (state >> i) & 1
            sj = (state >> (i + 1)) & 1

            # Sz_i Sz_j contribution (diagonal)
            H[idx, idx] += J * (0.25 if si == sj else -0.25)

            # S+_i S-_j + S-_i S+_j (off-diagonal: flip opposite spins)
            if si != sj:
                flipped = state ^ (1 << i) ^ (1 << (i + 1))
                jdx = state_to_idx.get(flipped)
                if jdx is not None:
                    H[idx, jdx] += J * 0.5  # (S+_i S-_j + h.c.) gives 1/2

    return H, basis

# Example: L=10, Sz=0 sector
L = 10
H_sz0, basis_sz0 = build_hamiltonian_sz_sector(L, sz_target=0)
eigvals = np.linalg.eigvalsh(H_sz0)
print(f"Sz=0 sector dimension: {H_sz0.shape[0]}")
print(f"(Full space dimension: {2**L})")
print(f"Ratio: {H_sz0.shape[0] / (2**L):.3f}")
print(f"Ground state energy: {eigvals[0]:.6f}")
```

## 2. Momentum Symmetry (Translation)

For periodic boundary conditions, the Hamiltonian is translationally invariant. We can block-diagonalize by total momentum $k = 2\pi n / L$.

### Translation Operator

```python
def translation_operator(L, basis):
    """
    Construct the translation operator T in a given basis.
    T|s_0, s_1, ..., s_{L-1}> = |s_{L-1}, s_0, ..., s_{L-2}>
    """
    dim = len(basis)
    T = np.zeros((dim, dim), dtype=float)
    state_to_idx = {state: i for i, state in enumerate(basis)}

    for idx, state in enumerate(basis):
        # Rotate bits right by 1 (periodic)
        lsb = state & 1
        translated = (state >> 1) | (lsb << (L - 1))
        jdx = state_to_idx.get(translated)
        if jdx is not None:
            T[idx, jdx] = 1.0

    return T
```

### Projection onto Momentum Sectors

```python
def momentum_sector_basis(L, basis, k):
    """
    Generate symmetry-adapted basis states for given momentum k = 2π n/L.
    Uses projector P_k = (1/L) Σ_{r=0}^{L-1} e^{ikr} T^r
    """
    dim = len(basis)
    state_to_idx = {state: i for i, state in enumerate(basis)}

    # Build translation matrix
    T = np.zeros((dim, dim), dtype=complex)
    for idx, state in enumerate(basis):
        lsb = state & 1
        translated = (state >> 1) | (lsb << (L - 1))
        jdx = state_to_idx.get(translated)
        if jdx is not None:
            T[idx, jdx] = 1.0

    # Build projector
    P = np.zeros((dim, dim), dtype=complex)
    T_power = np.eye(dim, dtype=complex)
    for r in range(L):
        P += np.exp(1j * k * r) * T_power
        T_power = T_power @ T

    P /= L

    # Extract basis states from the range of P
    _, indices = np.where(np.abs(P) > 1e-10)

    # Identify independent basis states via QR
    Q, R = np.linalg.qr(P)
    rank = np.sum(np.abs(np.diag(R)) > 1e-10)
    momentum_basis = Q[:, :rank]

    return momentum_basis, P
```

### Full Block Diagonalization

```python
def build_hamiltonian_momentum_sector(L, J=1.0, k=0, sz_target=0):
    """
    Build Hamiltonian in a combined Sz and momentum sector.
    """
    # First restrict to Sz sector
    H_sz, basis_sz = build_hamiltonian_sz_sector(L, J, sz_target)
    if H_sz is None:
        return None

    # Then project to momentum sector
    mom_basis, P = momentum_sector_basis(L, basis_sz, k)

    # H_k = P^† H P
    H_k = P.conj().T @ H_sz @ P

    # Project to the independent subspace
    Q, R = np.linalg.qr(H_k)
    rank = np.sum(np.abs(np.diag(R)) > 1e-10)
    H_reduced = H_k[:rank, :rank]

    return H_reduced

# Example: L=12, k=0, Sz=0
L = 12
H_k0 = build_hamiltonian_momentum_sector(L, k=0, sz_target=0)
if H_k0 is not None:
    eigvals = np.linalg.eigvalsh(H_k0)
    print(f"Momentum k=0 sector dimension: {H_k0.shape[0]}")
    print(f"Original Sz=0 sector: {H_k0.shape[0] * 4:.0f} (estimated)")
    print(f"Full Hilbert space: {2**L}")
    print(f"Ground state energy (k=0, Sz=0): {eigvals[0]:.6f}")
```

## 3. Performance Comparison

```python
def compare_sectors(L):
    """Compare dimensions across different symmetry sectors"""
    full_dim = 2 ** L
    sz0_dim = len(sz_sector_basis(L, 0)[0])

    # Momentum sectors share the Sz=0 space
    print(f"L={L}:")
    print(f"  Full space:         {full_dim:8d}")
    print(f"  Sz=0:               {sz0_dim:8d}  (ratio={sz0_dim/full_dim:.3f})")
    print(f"  Sz=0, k=0 (est.):  {sz0_dim//L:8d}  (ratio={sz0_dim/(L*full_dim):.3f})")

compare_sectors(12)
```

## Summary

- **$S^z$ symmetry**: Reduces dimension from $2^L$ to $\binom{L}{L/2}$ (roughly $1/\sqrt{\pi L/2}$ of the full space)
- **Momentum symmetry**: Further reduces each $S^z$ sector by approximately a factor of $L$
- Combined, we can reach $L = 16-20$ for spin-1/2 chains instead of $L \sim 12-14$ with naive ED
- For the $k=0$ sector, we need to handle the null space of the projector carefully via QR decomposition
