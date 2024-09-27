import numpy as np
import matplotlib.pyplot as plt

def pot_fn(r):
    if r == 0 : 
      return 0
    return -(e**2) / r

# Define Constant parameters
e = 3.795  # in eV*Å**(1/2)
h_cut = 1973.0  # (eV*A)  
m = 0.511e6  # (eV/c^2)

# Define System parameters
r0 = 0.00 
rn = 10.0  
N = 400  # number of points
d = (rn - r0) / (N)  # Step Size
R = np.linspace(r0+d, rn, N)  # Radial positions

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
    plt.plot(R, eVec[:,z[i]],label=f"n={i+1}, r={R[i]}", lw=1)
    plt.xlabel("r (Å)")
    plt.ylabel("Wave Function")
    plt.legend()
plt.suptitle("s-Wave Schrodinger's Equation")

plt.figure(figsize=(10,6))
for i in range(10):
    plt.axhline(energies[i], color='r')
plt.ylabel("Energy values")
plt.legend()
plt.show()