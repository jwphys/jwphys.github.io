---
layout: post
title: "Ladder Model — SSE & ED Implementation Notes"
date: 2026-05-17
tags: [python, fortran, sse-qmc, exact-diagonalization, spin-ladder]
lang: en
---

This note documents the implementation of the two-leg Heisenberg spin ladder model using both Stochastic Series Expansion (SSE) QMC and Exact Diagonalization (ED).

<div class="md-preview-container">
  <div class="md-preview-toolbar">
    <span class="md-preview-label">📄 Ladder_model.md</span>
    <div class="md-preview-actions">
      <a href="{{ '/assets/downloads/Ladder_model.md' | relative_url }}" class="md-download-btn" download>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Download
      </a>
    </div>
  </div>
  <div class="md-preview-scroll">
    <div class="md-preview-content">

## Ladder model

Ladder model describe two coupled Heisenberg chains. In each chain, spins are coupled with strength $J_\text{leg}$. And the two chain are coupled with strength $J_\text{rung}$. The Hamiltonian is

$$
H = J_\text{leg} \sum_{\alpha=0,1} \sum_{\langle i,j \rangle} \vec{S}_{\alpha,i} \cdot \vec{S}_{\alpha,j} + J_\text{rung}\sum_{i=0}^{L-1} \vec{S}_{0,i}\cdot\vec{S}_{1,i}
$$

We also consider PBC along each chain (x-direction), but not in y-direction.

### Achievement in SSE

Since we only have two chains, $L_x = L$, $L_y = 2$, total spins $N = 2L$. In PBC, number of bonds $N_b = N + L$, where $N$ is bonds in x-direction and $L$ in y-direction.

The `jr` variable in the configuration module:

```fortran
real(8) :: jr = 1.732d0      ! J2/J1 ratio
```

Modified `diagonalupdate`:

```fortran
subroutine diagonalupdate()
  use configuration; implicit none
  integer :: i,b,op
  real(8) :: ran, p
  external :: ran

  do i=0,mm-1
    op=opstring(i)
    if (op==0) then
      b=int(ran()*nb)+1
      if (spin(bsites(1,b))/=spin(bsites(2,b))) then
        p = aprob
        if (b>nn) p = p*jr   ! rung bond
        if (ran()*(mm-nh)<=p) then
          opstring(i)=2*b; nh=nh+1
        endif
      endif
    elseif (mod(op,2)==0) then
      b = op/2
      p = dprob
      if (b>nn) p = p/jr
      if (ran()<=p*(mm-nh+1)) then
        opstring(i)=0; nh=nh-1
      endif
    else
      b=op/2
      spin(bsites(1,b))=-spin(bsites(1,b))
      spin(bsites(2,b))=-spin(bsites(2,b))
    endif
  enddo
end subroutine diagonalupdate
```

### Achievement in ED

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
sy = np.array([[0,-1j],[1j, 0]], dtype=complex) / 2
sz = np.array([[1, 0], [0,-1]], dtype=complex) / 2
I2 = np.eye(2, dtype=complex)

def Hamiltonian_sparse(L, J1=1, J2=1.732):
    N = 2 * L
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    def two_site(N, i, j, oi, oj):
        ops = [oi if s==i else oj if s==j else I2 for s in range(N)]
        M = sp.csr_matrix(ops[0])
        for m in ops[1:]: M = sp.kron(M, sp.csr_matrix(m), format="csr")
        return M
    for k in [0,1]:
        for i in range(L):
            j = (i+1)%L
            si, sj = k*L+i, k*L+j
            for o in [(sx,sx),(sy,sy),(sz,sz)]:
                H += J1 * two_site(N, si, sj, o[0], o[1])
    for i in range(L):
        for o in [(sx,sx),(sy,sy),(sz,sz)]:
            H += J2 * two_site(N, i, L+i, o[0], o[1])
    return H.tocsr()
```

### Results

```
ED correlation function:
[[ 0.25     -0.183845 -0.148214  0.082059]
 [-0.183845  0.25      0.082059 -0.148214]
 [-0.148214  0.082059  0.25     -0.183845]
 [ 0.082059 -0.148214 -0.183845  0.25    ]]
```

    </div>
  </div>
</div>
