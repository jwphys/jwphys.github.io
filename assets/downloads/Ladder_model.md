# Ladder model

Ladder model describe two coupled Heisenberg chains.  In each chain, spins are coupled with strength $J_\text{leg}$. And the two chain are coupled with strength $J_\text{rung}$. The Hamiltonian is 

$$
H = J_\text{leg} \sum_{\alpha=0,1} \sum_{<i,j>} \vec{S}_{\alpha,i} \cdot \vec{S}_{\alpha,j} + J_\text{rung}\sum_{i=0}^{L-1} \vec{S}_{0,i}\cdot\vec{S}_{1,i}
$$

where $\alpha$ is indicates which chain it is and $i,j$ locate in chain direction. 

```
## O represent spin. J1 and J2 are J_leg and J_rung, respectively.
O ——J1—— O ——J1—— O ——J1—— O  
|        |        |        | 
J2       J2       J2       J2 
|        |        |        | 
O ——J1—— O ——J1—— O ——J1—— O  
```

We also consider PBC along each chain (upper or lower chain, in x-direction). We don't push PBC in y-direction. 

## Achievement in SSE

Since we only have two chains, which means $L_x=L,L_y=2$, the total number of spins is $N=2L$. In PBC, we will have the number of bonds of $N_b=N+L$, where $N$ is number of bonds in x-direction and $L$ is number of bonds in y-direction. In this way, when we scan bond, if the iterative variable is less than $N$, the interaction is $J_1$ and the else is $J_2$.

When put this rule to diagonalization process, we will get the ladder model in standard Heisenberg model. Because  the off-diagonal operators are replaced from diagonal operators, when we modify interaction strength in diagonal operators, the interaction strength of off-diagonal operators change was modified.  

```
## more clear explanation:
In order to achieve such model according to SSE algorithm, we could modify the probability of acceptance/delete of diagonal operators. We don't need to modify that for off-diagonal operators as the off-diagonal operators are flipped from diagonal operators through loop update. In such case, we would introduce the different strength for bonds. 
```

Allocate the `jr` variable in `configuration` module

```fortran
 real(8) :: jr = 1.732d0      ! J2/J1 ratio, ladder model    
```

Modify `diagonalupdate` module

```fortran
!---------------------------!
 subroutine diagonalupdate()
!------------------------------------------!
! Carries out one sweep of diagonal updates
!------------------------------------------!
 use configuration; implicit none

 integer :: i,b,op
 real(8) :: ran
 real(8) :: p

 external :: ran

 do i=0,mm-1
    op=opstring(i)
    if (op==0) then       
       b=int(ran()*nb)+1 ! random bond
       if (spin(bsites(1,b))/=spin(bsites(2,b))) then
         p = aprob ! insert probability
         if (b>nn) then ! the bond for rungs
            p = p*jr ! J2 probability for insertion
         endif
          if (ran()*(mm-nh)<=p) then
             opstring(i)=2*b
             nh=nh+1 
          endif
       endif
    elseif (mod(op,2)==0) then        
         b = op/2 ! bond index
         p = dprob ! drop probability
         if (b>nn) then 
            p = p/jr ! J1 probability for deletion
         endif
         if (ran()<=p*(mm-nh+1)) then
            opstring(i)=0
            nh=nh-1
         endif
    else
       b=op/2
       spin(bsites(1,b))=-spin(bsites(1,b))
       spin(bsites(2,b))=-spin(bsites(2,b))
    endif
 enddo

 end subroutine diagonalupdate
!-----------------------------!
```

## Achievement in ED

Compare to 1D chain's ED program, we have to introduce sparse matrix even in low dimension to storage such huge data. And also, the Hamiltonian is completely different compared to 1D chain. 

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Define spin-1/2 operators
sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
I2 = np.eye(2, dtype=complex)

def kron_n_sparse(ops):
    M = sp.csr_matrix(ops[0])
    for op in ops[1:]:
        M = sp.kron(M, sp.csr_matrix(op), format="csr")
    return M

def two_site_op_sparse(N, i, j, op_i, op_j):
    ops = []
    for site in range(N):
        if site == i:
            ops.append(op_i)
        elif site == j:
            ops.append(op_j)
        else:
            ops.append(I2)
    return kron_n_sparse(ops)

def Hamiltonian_sparse(L, J1=1, J2=1.732):
    N = 2 * L
    H = sp.csr_matrix((2**N, 2**N), dtype=complex)
    # leg couplings
    for k in [0, 1]:
        for i in range(L):
            j = (i + 1) % L
            site_i = k * L + i
            site_j = k * L + j
            for op in [(sx, sx), (sy, sy), (sz, sz)]:
                H += J1 * two_site_op_sparse(N, site_i, site_j, op[0], op[1])
    # rung couplings
    for i in range(L):
        site_i = i
        site_j = L + i
        for op in [(sx, sx), (sy, sy), (sz, sz)]:
            H += J2 * two_site_op_sparse(N, site_i, site_j, op[0], op[1])
    return H.tocsr()

def get_nth_eigenstate(H, n):
    E, V = spla.eigsh(H)
    idx = np.argsort(E)
    E = E[idx]; V = V[:, idx]
    return E[n], V[:, n]

def SzSz_correlation(N, psi_n):
    C_zz = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                C_zz[i, j] = 0.25
            else:
                SzSz_ij = two_site_op_sparse(N, i, j, sz, sz)
                C_zz[i, j] = np.vdot(psi_n, SzSz_ij @ psi_n)
    return C_zz

# example
L = 2; N = 2*L; J1 = 1; J2 = 1.732
H = Hamiltonian_sparse(L, J1, J2)
E_n, psi_n = get_nth_eigenstate(H, 0)
C_zz = SzSz_correlation(N, psi_n)
print(f"Ground state energy: {E_n.real:.6f}")
print("C_zz (real part):")
print(np.round(C_zz.real, 6))
```

## Result

correlation function:

```
ED:
[[ 0.25     -0.183845 -0.148214  0.082059]
 [-0.183845  0.25      0.082059 -0.148214]
 [-0.148214  0.082059  0.25     -0.183845]
 [ 0.082059 -0.148214 -0.183845  0.25    ]]

SSE:
0.25000000000000000      -0.18590000000000001      -0.14920000000000000        8.5099999999999995E-002
-0.18590000000000001       0.25000000000000000        8.5099999999999995E-002 -0.14920000000000000     
-0.14920000000000000        8.5099999999999995E-002  0.25000000000000000      -0.18590000000000001     
  8.5099999999999995E-002 -0.14920000000000000      -0.18590000000000001       0.25000000000000000     
```
