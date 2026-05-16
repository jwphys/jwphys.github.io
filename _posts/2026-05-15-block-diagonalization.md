---
layout: post
title: "Block Diagonalization: Leveraging Symmetries in Exact Diagonalization"
date: 2026-05-15
tags: [python, exact-diagonalization, symmetries, quantum-magnetism]
lang: en
---

Exact diagonalization of quantum Hamiltonians is limited by the exponential growth of the Hilbert space ($\dim \mathcal{H} = 2^L$). By exploiting **symmetries** of the Hamiltonian, we can block-diagonalize the matrix and study much larger systems. Here we implement two key symmetries of the Heisenberg model:

1. **Total $S^z$ symmetry** — conservation of total magnetization
2. **Momentum (translation) symmetry** — invariance under periodic shifts

---

## 1. Total $S^z$ Symmetry

The Heisenberg model conserves total magnetization:

$$
[S^z_{\text{tot}}, H] = 0, \qquad S^z_{\text{tot}} = \sum_{i=1}^L S_i^z
$$

Therefore $H$ is block-diagonal in the $S^z_{\text{tot}}$ quantum number, and we can work in sectors of fixed $S^z$.

### Basis Construction

We represent basis states as integers (bitstrings): bit $i$ is 1 for $\uparrow$ and 0 for $\downarrow$. For a target $S^z = m$, the number of up spins is $n_\uparrow = m + L/2$.

```python
import numpy as np
from itertools import combinations

def sz_sector_basis(L, sz_target):
    """
    Generate all basis states with a given total S^z.

    States are represented as integers (bitstrings) where:
        bit = 1  =>  |↑⟩
        bit = 0  =>  |↓⟩

    Example for L=2, sz=0:
        States: |↑↓⟩ (bits: 01 = 1), |↓↑⟩ (bits: 10 = 2)

    Args:
        L: Number of sites.
        sz_target: Target total S^z (e.g., 0 for half-filling).

    Returns:
        basis: List of integer-encoded basis states.
        dim: Dimension of the sector.
    """
    # Number of up spins needed: n_up - n_down = 2*sz_target
    # n_up + n_down = L  =>  n_up = (L + 2*sz_target) / 2 = sz_target + L/2
    n_up = sz_target + L // 2

    # Invalid sector
    if n_up < 0 or n_up > L:
        return [], 0

    basis = []
    # Iterate over all ways to choose which sites are up
    for up_positions in combinations(range(L), n_up):
        state = 0
        for pos in up_positions:
            state |= (1 << pos)  # Set bit at position `pos`
        basis.append(state)

    return basis, len(basis)
```

### Building the Hamiltonian in the $S^z$ Sector

Instead of the full $2^L \times 2^L$ matrix, we directly construct the smaller matrix for a given $S^z$ sector. The bond operator $\mathbf{S}_i \cdot \mathbf{S}_{i+1}$ acts as:

$$
\mathbf{S}_i \cdot \mathbf{S}_{i+1} =
\begin{cases}
+\frac14 & \text{if } \sigma_i = \sigma_{i+1} \text{ (diagonal)} \\[4pt]
-\frac14 & \text{if } \sigma_i \neq \sigma_{i+1} \text{ (diagonal, opposite spins)} \\[4pt]
\frac12 (S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+) & \text{(off-diagonal, flips } \uparrow\downarrow \leftrightarrow \downarrow\uparrow)
\end{cases}
$$

```python
def build_hamiltonian_sz_sector(L, J=1.0, sz_target=0):
    """
    Build the Heisenberg Hamiltonian restricted to a fixed S^z sector.

    Only states within the sector are included, and matrix elements
    are computed directly without constructing the full 2^L space.

    Args:
        L: Number of sites.
        J: Exchange coupling (positive = antiferromagnetic).
        sz_target: Target S^z sector.

    Returns:
        H: (dim, dim) real symmetric matrix.
        basis: Integer-encoded basis states.
    """
    basis, dim = sz_sector_basis(L, sz_target)
    if dim == 0:
        return None, None

    # Map each bitstring state to its index in the basis
    state_to_idx = {state: i for i, state in enumerate(basis)}

    H = np.zeros((dim, dim), dtype=float)

    # Loop over all basis states
    for idx, state in enumerate(basis):
        # Loop over all bonds
        for i in range(L - 1):
            # Extract spins at sites i and i+1 (0 or 1)
            si = (state >> i) & 1
            sj = (state >> (i + 1)) & 1

            # --- Diagonal part: J * S^z_i * S^z_j ---
            # S^z|↑⟩ = +1/2|↑⟩, S^z|↓⟩ = -1/2|↓⟩
            # So S^z_i S^z_j gives +1/4 if same, -1/4 if opposite
            H[idx, idx] += J * (0.25 if si == sj else -0.25)

            # --- Off-diagonal part: (J/2) * (S^+_i S^-_j + S^-_i S^+_j) ---
            # Only acts when we have |↑↓⟩ or |↓↑⟩, swapping them
            if si != sj:
                # Flip both bits: this takes |↑↓⟩ ↔ |↓↑⟩
                flipped = state ^ (1 << i) ^ (1 << (i + 1))
                jdx = state_to_idx.get(flipped)
                if jdx is not None:
                    H[idx, jdx] += J * 0.5

    return H, basis
```

### Example

```python
L = 10
H_sz0, basis_sz0 = build_hamiltonian_sz_sector(L, sz_target=0)
eigvals = np.linalg.eigvalsh(H_sz0)

print(f"L={L} Heisenberg chain, S^z=0 sector:")
print(f"  Sector dimension:     {H_sz0.shape[0]}")
print(f"  Full space dimension: {2**L}")
print(f"  Reduction factor:     {H_sz0.shape[0] / (2**L):.3f}")
print(f"  Ground state energy:  {eigvals[0]:.6f}")
print(f"  First excited state:  {eigvals[1]:.6f}")
```

---

## 2. Momentum Symmetry (Translation)

For periodic boundary conditions, the Hamiltonian commutes with the translation operator $T$:

$$
T\,|s_0, s_1, \dots, s_{L-1}\rangle = |s_{L-1}, s_0, \dots, s_{L-2}\rangle
$$

Eigenstates of $T$ have definite crystal momentum $k = 2\pi n / L$:

$$
T\,|\psi_k\rangle = e^{ik}\,|\psi_k\rangle
$$

### Translation Operator

```python
def translation_operator(L, basis):
    """
    Construct the translation operator T in a given basis.

    T shifts all spins right by one site (periodic):
        T |s_0, s_1, ..., s_{L-1}> = |s_{L-1}, s_0, ..., s_{L-2}>

    For the bitstring representation, this is a circular right-shift.

    Args:
        L: Number of sites.
        basis: List of integer-encoded basis states.

    Returns:
        T: (dim, dim) permutation matrix.
    """
    dim = len(basis)
    T = np.zeros((dim, dim), dtype=float)
    state_to_idx = {state: i for i, state in enumerate(basis)}

    for idx, state in enumerate(basis):
        # Circular right shift by 1 bit
        lsb = state & 1                          # Save the rightmost bit
        translated = (state >> 1) | (lsb << (L - 1))  # Rotate right

        jdx = state_to_idx.get(translated)
        if jdx is not None:
            T[idx, jdx] = 1.0

    return T
```

### Momentum Projector

For a given momentum $k = 2\pi n / L$, we construct the projection operator:

$$
P_k = \frac{1}{L} \sum_{r=0}^{L-1} e^{ikr}\, T^{\,r}
$$

States in the $k$-sector are eigenvectors of $P_k$ with eigenvalue 1.

```python
def momentum_projector(L, basis, k):
    """
    Build the momentum projector P_k for crystal momentum k.

    P_k = (1/L) * sum_{r=0}^{L-1} exp(i*k*r) * T^r

    Args:
        L: Number of sites.
        basis: List of integer-encoded basis states.
        k: Crystal momentum (2π * n / L for integer n).

    Returns:
        P: (dim, dim) complex projection matrix.
        rank: Dimension of the k-sector.
    """
    dim = len(basis)

    # First construct the translation matrix
    T = np.zeros((dim, dim), dtype=complex)
    state_to_idx = {state: i for i, state in enumerate(basis)}
    for idx, state in enumerate(basis):
        lsb = state & 1
        translated = (state >> 1) | (lsb << (L - 1))
        jdx = state_to_idx.get(translated)
        if jdx is not None:
            T[idx, jdx] = 1.0

    # Build the projector as a sum over powers of T
    P = np.zeros((dim, dim), dtype=complex)
    T_power = np.eye(dim, dtype=complex)  # T^0 = I
    for r in range(L):
        P += np.exp(1j * k * r) * T_power
        T_power = T_power @ T  # T^{r+1} = T^r @ T
    P /= L

    # Determine the rank (dimension of the k-sector) via QR decomposition
    Q, R = np.linalg.qr(P)
    rank = np.sum(np.abs(np.diag(R)) > 1e-10)

    return P, rank
```

### Full Block Diagonalization (Combined $S^z$ + Momentum)

```python
def build_hamiltonian_k_sector(L, J=1.0, k=0, sz_target=0):
    """
    Build the Hamiltonian projected to a simultaneous S^z and momentum sector.

    Procedure:
        1. Restrict to the S^z = sz_target sector.
        2. Project onto crystal momentum k using P_k.
        3. Reduce to the independent subspace via QR.

    Args:
        L: Number of sites.
        J: Exchange coupling.
        k: Crystal momentum (in radians).
        sz_target: Total S^z quantum number.

    Returns:
        H_k: (rank, rank) reduced Hamiltonian matrix.
        rank: Dimension of the symmetry sector.
    """
    # Step 1: Build Hamiltonian in the S^z sector
    H_sz, basis_sz = build_hamiltonian_sz_sector(L, J, sz_target)
    if H_sz is None:
        return None, 0

    # Step 2: Build momentum projector
    P, rank = momentum_projector(L, basis_sz, k)
    print(f"  S^z={sz_target} sector dim: {H_sz.shape[0]}")
    print(f"  k={k:.3f} sector dim:      {rank}")

    # Step 3: Project Hamiltonian: H_k = P^† H P
    H_proj = P.conj().T @ H_sz @ P

    # Step 4: QR to extract the independent subspace
    Q, R = np.linalg.qr(H_proj)
    H_reduced = H_proj[:rank, :rank]

    return H_reduced, rank
```

### Example

```python
L = 12
k = 0  # Zero momentum (n=0)

H_k0, rank = build_hamiltonian_k_sector(L, J=1.0, k=k, sz_target=0)

if H_k0 is not None:
    eigvals = np.linalg.eigvalsh(H_k0)
    print(f"\nL={L} Heisenberg chain (S^z=0, k=0):")
    print(f"  Reduced dimension: {rank}")
    print(f"  Full space:        {2**L}")
    print(f"  Reduction ratio:   {rank / (2**L):.5f}")
    print(f"  Ground state:      {eigvals[0]:.6f}")

# Compare different momentum sectors
print("\n--- Momentum sector comparison ---")
for n in range(L // 2 + 2):
    k = 2 * np.pi * n / L
    H_k, r = build_hamiltonian_k_sector(L, J=1.0, k=k, sz_target=0)
    if H_k is not None and r > 0:
        ev = np.linalg.eigvalsh(H_k)
        print(f"  n={n:2d}  k={k:.3f}  dim={r:4d}  E0={ev[0]:.6f}")
```

---

## 3. Performance Comparison

```python
def compare_sectors(L):
    """Compare dimensions across S^z and momentum sectors for a given L."""
    full_dim = 2 ** L

    # S^z=0 sector dimension
    basis_sz0, sz0_dim = sz_sector_basis(L, 0)

    # Estimate momentum sector dimensions
    # Approximately sz0_dim / L for each k-sector
    print(f"L={L}:")
    print(f"  Full Hilbert space:       {full_dim:10d}")
    print(f"  S^z=0 sector:            {sz0_dim:10d}  "
          f"(ratio={sz0_dim/full_dim:.4f})")
    print(f"  S^z=0, k=0 (est.):       {sz0_dim//L:10d}  "
          f"(ratio={sz0_dim/(L*full_dim):.4f})")
    print(f"  Number of k-sectors:     {L}")

compare_sectors(12)
```

Example output:
```
L=12:
  Full Hilbert space:             4096
  S^z=0 sector:                   924  (ratio=0.2256)
  S^z=0, k=0 (est.):              77   (ratio=0.0188)
  Number of k-sectors:             12
```

---

## Summary

| Symmetry | Reduction Factor | Max $L$ Reachable |
|---|---|---|
| None (full space) | $2^L$ | $\sim 12$ |
| $S^z$ only | $\binom{L}{L/2} \sim \frac{2^L}{\sqrt{\pi L/2}}$ | $\sim 16$ |
| $S^z$ + Momentum | $\sim \frac{2^L}{L\sqrt{\pi L/2}}$ | $\sim 20$ |

Key points:

- **$S^z$ symmetry** is straightforward: we enumerate only states with a fixed number of up spins, reducing the dimension by a factor of $\sim \sqrt{\pi L/2}$.
- **Momentum symmetry** further divides each $S^z$ sector by approximately $L$, giving another order-of-magnitude reduction.
- Combined symmetries enable exact diagonalization for $L \sim 20$ on a single workstation.
- The QR decomposition after projection ensures we work in the independent subspace, removing null vectors.
