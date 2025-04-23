#%%
import numpy as np
from scipy import sparse as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d 

#Defining variables
N =100 #Discretization
u0_t, u0_b = 5, -5 #Boundary condition along y-axis
u0_l, u0_r = -1, 3 #Boundary condition along x-axis
dx = 1./(N+1) #Step-size assuming our geometry is a square of length 1m

#Constructing the 1D discretization matrix
A_1D = (sp.eye(N, k=-1) - 2*sp.eye(N, k=0) + sp.eye(N, k=1))/ dx**2

I = sp.eye(N) #The Identity matrix of shape (N x N)

#Constructing the 2D discretization matrix using tensor product (N**2 x N**2)
A = sp.kron(I, A_1D) + sp.kron(A_1D, I)

#Heating source term
H_t = -5

#Applying boundary conditions (Dirichlet BCs)
b = H_t * np.ones((N, N))
b[0,:] = H_t - (u0_b/dx**2) #BC at y=0
b[-1,:] = H_t - (u0_t/dx**2) #BC at y=1
b[:, 0] = H_t - (u0_r/dx**2) #BC at x=0
b[:, -1] = H_t - (u0_l/dx**2) #BC at x=1
b = b.reshape(N**2) 

#Solving the linear system
u_row_major = sp.linalg.spsolve(A, b) 
u = u_row_major.reshape(N, N)

U = np.vstack([
    np.ones((1, N+2)) * u0_b,                             # Bottom boundary (1 row)
    np.hstack([                                           # Middle block (N rows)
        np.ones((N, 1)) * u0_l,                           # Left boundary (Nx1)
        u,                                                # Interior values (NxN)
        np.ones((N, 1)) * u0_r                            # Right boundary (Nx1)
    ]),
    np.ones((1, N+2)) * u0_t                              # Top boundary (1 row)
    ])
x = np.linspace(0, 1, N+2)
X, Y = np.meshgrid(x, x)
fig = plt.figure(figsize=(12, 5.5))
cmap = mpl.cm.get_cmap('jet')

ax = fig.add_subplot(1, 2, 1)
c = ax.pcolor(X, Y, U, vmin=-5, vmax=5, cmap=cmap)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)

ax = fig.add_subplot(1, 2, 2, projection='3d')
p = ax.plot_surface(X, Y, U, vmin=-5, vmax=5, rstride=3, cstride=3, linewidth=0, cmap=cmap)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
cb = plt.colorbar(p, ax=ax, shrink=0.75)
cb.set_label(r"$u(x_1, x_2)$", fontsize=18)
plt.show()
# %%
