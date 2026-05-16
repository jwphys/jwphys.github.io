---
layout: post
title: "Exact Diagonalization of 1D Spin Chains and Ladder Models in Python"
date: 2026-05-16
tags: [python, exact-diagonalization, spin-chain, quantum-magnetism]
lang: en
---

Exact diagonalization (ED) is a powerful numerical method for studying finite quantum many-body systems. By constructing and diagonalizing the Hamiltonian matrix in a chosen basis, we obtain exact ground-state energies, excited states, and correlation functions without any approximation (up to numerical precision).

---

## 1. Spin-1/2 Heisenberg Chain

The Hamiltonian for the 1D spin-$1/2$ Heisenberg chain with periodic boundary conditions:

$$
H = J \sum_{i=1}^{L-1} \mathbf{S}_i \cdot \mathbf{S}_{i+1}
$$

We work in the $S^z$ eigenbasis: $|\sigma_1, \sigma_2, \dots, \sigma_L\rangle$, where each $\sigma_i = \uparrow$ or $\downarrow$. The dimension of the Hilbert space is $2^L$.

### Pauli and Spin Operators

```python
import numpy as np

def pauli_matrices():
    """
    Return the three Pauli matrices as 2x2 complex arrays.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return sx, sy, sz

def spin_operators(S=0.5):
    """
    Return spin-S operators: S^x, S^y, S^z as 2x2 matrices.
    For spin-1/2, S^a = (1/2) * sigma^a.
    """
    sx, sy, sz = pauli_matrices()
    return S * sx, S * sy, S * sz
```

### Building the Hamiltonian

We construct the full $2^L \times 2^L$ Heisenberg Hamiltonian via Kronecker products. Each bond term $\mathbf{S}_i \cdot \mathbf{S}_{i+1}$ is built by placing spin operators at sites $i$ and $i+1$, and identity matrices everywhere else.

```python
def heisenberg_chain(L, J=1.0):
    """
    Build the Heisenberg chain Hamiltonian H = J * sum_{i} S_i · S_{i+1}.

    Args:
        L: Number of sites.
        J: Exchange coupling (J>0 is antiferromagnetic).

    Returns:
        H: (2^L, 2^L) complex Hermitian matrix.
    """
    dim = 2 ** L
    sx, sy, sz = spin_operators(0.5)
    eye2 = np.eye(2, dtype=complex)

    H = np.zeros((dim, dim), dtype=complex)

    for i in range(L - 1):
        # Build a list of operators for each site
        before = [eye2] * i          # sites 0 .. i-1
        after  = [eye2] * (L - i - 2) # sites i+2 .. L-1

        # For the bond (i, i+1), construct X_i X_{i+1}
        ops_x = before + [sx, sx] + after
        ops_y = before + [sy, sy] + after
        ops_z = before + [sz, sz] + after

        def kron_product(op_list):
            """Kronecker product of a list of matrices in order."""
            result = op_list[0]
            for m in op_list[1:]:
                result = np.kron(result, m)
            return result

        H += J * (kron_product(ops_x) + kron_product(ops_y) + kron_product(ops_z))

    return H
```

### Example: Compute the Ground State

```python
L = 8
H = heisenberg_chain(L)

# Full diagonalization (L <= 12)
eigvals, eigvecs = np.linalg.eigh(H)

# Ground state
gs_energy = eigvals[0]
gs_wavefunction = eigvecs[:, 0]

print(f"L={L} Heisenberg chain:")
print(f"  Ground state energy:       {gs_energy:.6f}")
print(f"  First excited state:       {eigvals[1]:.6f}")
print(f"  Gap to first excitation:   {eigvals[1] - eigvals[0]:.6f}")
```

For larger systems ($L > 12$), use sparse diagonalization:

```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def heisenberg_chain_sparse(L, J=1.0):
    """Build sparse Heisenberg Hamiltonian (more efficient for large L)."""
    from scipy.sparse import kron as sparse_kron
    from scipy.sparse import eye as speye

    sx, sy, sz = spin_operators(0.5)
    eye2 = np.eye(2, dtype=complex)

    H = None
    for i in range(L - 1):
        # Build sparse bond term
        bond = np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)
        bond_sp = csr_matrix(bond)

        # Place bond at (i, i+1) via sparse Kronecker products
        if i == 0:
            term = csr_matrix(bond)
        else:
            term = csr_matrix(eye2)
        for j in range(1, L - 1):
            left = csr_matrix(eye2) if j != i else csr_matrix(bond)
            term = sparse_kron(term, left)

        if H is None:
            H = J * term
        else:
            H += J * term

    return H

# Sparse diagonalization for L=14, request 5 lowest eigenvalues
L = 14
H_sp = heisenberg_chain_sparse(L)
eigvals_sp = eigsh(H_sp, k=5, which='SA', return_eigenvectors=False)
print(f"\nL={L} Heisenberg chain (sparse):")
print(f"  Ground state energy: {eigvals_sp[0]:.6f}")
```

---

## 2. Spin Ladder Model

A two-leg spin-1/2 ladder consists of $L$ rungs (2 spins per rung, $2L$ spins total):

$$
H = J_\parallel \sum_{\alpha=1,2} \sum_{i=1}^{L-1} \mathbf{S}_{i,\alpha} \cdot \mathbf{S}_{i+1,\alpha}
    + J_\perp \sum_{i=1}^{L} \mathbf{S}_{i,1} \cdot \mathbf{S}_{i,2}
$$

where $\alpha = 1, 2$ labels the two legs.

```python
def spin_ladder(L, J_parallel=1.0, J_perp=1.0):
    """
    Build two-leg spin ladder Hamiltonian.

    Leg 1 sites:  0,   1,   2,  ..., L-1
    Leg 2 sites:  L,  L+1, L+2, ..., 2L-1
    Rung bonds:  (0,L), (1,L+1), ..., (L-1, 2L-1)

    Args:
        L: Number of rungs.
        J_parallel: Intra-leg coupling.
        J_perp: Rung coupling.

    Returns:
        H: (2^{2L}, 2^{2L}) Hamiltonian matrix.
    """
    dim = 2 ** (2 * L)
    sx, sy, sz = spin_operators(0.5)
    eye2 = np.eye(2, dtype=complex)

    H = np.zeros((dim, dim), dtype=complex)

    def kron_at(ops_list, total_sites):
        """
        Build a Kronecker product over `total_sites` sites.
        `ops_list` is a list of (site_index, matrix) pairs.
        Any site not in ops_list gets the identity.
        """
        # Build a lookup dict
        op_dict = dict(ops_list)
        mats = [op_dict.get(i, eye2) for i in range(total_sites)]
        result = mats[0]
        for m in mats[1:]:
            result = np.kron(result, m)
        return result

    # --- Intra-leg couplings ---
    for leg in range(2):
        offset = leg * L
        for i in range(L - 1):
            bond_x = [(offset + i, sx), (offset + i + 1, sx)]
            bond_y = [(offset + i, sy), (offset + i + 1, sy)]
            bond_z = [(offset + i, sz), (offset + i + 1, sz)]
            H += J_parallel * (
                kron_at(bond_x, 2*L) +
                kron_at(bond_y, 2*L) +
                kron_at(bond_z, 2*L)
            )

    # --- Rung couplings ---
    for i in range(L):
        # Leg 1 site i, Leg 2 site i+L
        rung_x = [(i, sx), (i + L, sx)]
        rung_y = [(i, sy), (i + L, sy)]
        rung_z = [(i, sz), (i + L, sz)]
        H += J_perp * (
            kron_at(rung_x, 2*L) +
            kron_at(rung_y, 2*L) +
            kron_at(rung_z, 2*L)
        )

    return H
```

### Example: Compute the Spin Gap

The spin gap $\Delta = E_1 - E_0$ is the energy difference between the first excited state and the ground state. For a spin ladder, the gap distinguishes the rung-singlet phase ($\Delta > 0$) from the gapless phase.

```python
L_rung = 4
H_ladder = spin_ladder(L_rung, J_parallel=1.0, J_perp=0.5)
eigvals = np.linalg.eigvalsh(H_ladder)

print(f"Ladder (L={L_rung} rungs, J_∥={1.0}, J_⊥={0.5}):")
print(f"  Ground state energy: {eigvals[0]:.6f}")
print(f"  Spin gap:            {eigvals[1] - eigvals[0]:.6f}")
```

---

## 3. Measuring Observables

Once we have the ground-state wavefunction $|\psi_0\rangle$, we compute physical observables.

```python
def expectation_value(state, operator):
    """
    Compute <state|operator|state> for a given quantum state.
    """
    return state.conj().T @ operator @ state


def sz_correlation(state, L, i, j):
    """
    Compute the spin-spin correlation <S^z_i S^z_j>.

    Args:
        state: Ground state wavefunction (length 2^L).
        L: Number of sites.
        i, j: Site indices.

    Returns:
        <S^z_i S^z_j> as a real number.
    """
    sx, sy, sz = spin_operators(0.5)
    eye2 = np.eye(2, dtype=complex)

    # Build the operator S^z_i ⊗ S^z_j
    before_i = [eye2] * i
    between  = [eye2] * (j - i - 1)
    after_j  = [eye2] * (L - j - 1)

    op_parts = before_i + [sz] + between + [sz] + after_j

    op = op_parts[0]
    for m in op_parts[1:]:
        op = np.kron(op, m)

    return expectation_value(state, op).real


# Example: compute correlation function for L=8 chain
L = 8
H = heisenberg_chain(L)
eigvals, eigvecs = np.linalg.eigh(H)
psi0 = eigvecs[:, 0]

print(f"\nSpin-spin correlation <S^z_0 S^z_r> for L={L}:")
for r in range(1, L // 2 + 1):
    corr = sz_correlation(psi0, L, 0, r)
    print(f"  r={r}: {corr:.6f}")
```

---

## Summary

Exact diagonalization provides **exact** results for small systems (typically $L \lesssim 16$ for spin-1/2 chains, or $L \lesssim 12$ for ladders). It serves as a benchmark for approximate methods like DMRG and enables detailed study of:

- Energy spectra and gaps
- Correlation functions
- Entanglement entropy and spectra
- Finite-size scaling

The main limitation is the exponential growth of the Hilbert space. In the next note, we show how to exploit symmetries (total $S^z$ and momentum) to reach larger system sizes.
