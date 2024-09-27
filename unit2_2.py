import numpy as np
import matplotlib.pyplot as plt

def pot_fn(r):
    if (r<=0 or r>=a):
        return V0
    else:
        return 0

# Define Constant parameters
h_cut = 1973.0  # (eV*A)  
m = 0.511e6  # (eV/c^2)
V0=100

# Define System parameters
r0 = -5.00 
rn = 10.0  
a=5.0
N = 300  # number of points
d = (rn - r0) / (N+1)  # Step Size
R = np.linspace(r0, rn, N)  # Radial positions


K = np.zeros((len(R), len(R)))  # Kinetic energy matrix
V = np.zeros((len(R), len(R)))  # Potential energy matrix

for i in range(len(R)):
    for j in range(len(R)):
        if i == j:
            K[i,j] = -2
            V[i,j] = pot_fn(R[i])
        elif np.abs(i - j) == 1:
            K[i,j] = 1

# Scaling the kinetic energy matrix
K = (-(h_cut**2) / (2 * m * (d**2))) * K

# Constructing the Hamiltonian matrix
H = K + V

# Solve the eigenvalue equation
eVal, eVec = np.linalg.eig(H)
z = np.argsort(eVal)
energies = eVal[z]

# Print the first three energy levels in eV
print("First three energy levels (eV):", energies[:3])

#plot the first 4 eigen states 
plt.figure(figsize=(10,6))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(R, eVec[:,z[i]],label=f"n={i+1}, E{i+1}={energies[i]}", lw=1)
    plt.axvline(0, color='r',lw=.3,ymax=V0)
    plt.axvline(a, color='r',lw=.3,ymax=V0)
    plt.axhline(0, color='k',lw=.3)
    plt.xlabel("x (Å)")
    plt.ylabel("Wave Function")
    plt.legend()
plt.suptitle("Solution of 1-D Finite Potential Well")

plt.figure(figsize=(10,6))
for i in range(10):
    plt.axhline(energies[i], color='r')
plt.ylabel("Energy values")
plt.legend()
plt.show()
