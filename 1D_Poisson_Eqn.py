#%%
import numpy as np
from scipy import interpolate
import scipy.linalg as la
import matplotlib.pyplot as plt

N = 5 #Discretization
u0, u1 = 1, 2 #Boundary Conditions
dx = 1/(N+1) #Grid spacing

#Construct the system matrix A (Laplacian operator using finite differences)
A = (np.eye(N, k=-1) - 2*np.eye(N, k=0) + np.eye(N, k=1))/(dx**2) 

#Heating source term
H_t = -5
#Define the right-hand side vector b for Poisson's equation: u_xx = -5
b = H_t * np.ones(N) #RHS values for interior points (source term is -5)

#Modify b to include the effect of Dirichlet boundary conditions
b[0] = H_t - u0/dx**2 #Influence of u(0)
b[-1] = H_t - u1/dx**2 #Influence of u(1)

#Solve the linear system
u = la.solve(A, b)

x = np.linspace(0, 1, N+2)
U = np.hstack([[u0], u, [u1]]) #Include boundary values to get the full solution
U_x = interpolate.interp1d(x, U, axis=0, kind='cubic') #Interpolate for smooth plotting

#Plot the solution
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(x, U)
ax.plot(x[1:-1], u, 'ks')
ax.set_xlim(0, 1)
ax.set_xlabel(r'$x$', fontsize=18)
ax.set_ylabel(r'$U(x)$', fontsize=18)
plt.show()
# %%
