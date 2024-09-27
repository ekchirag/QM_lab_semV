
import numpy as np
import matplotlib.pyplot as plt

def pot_fn(r):
    return ((k*r**2)/2)+((b*r**3)/3)

# Define Constant parameters
h_cut = 197.30  # (MeVfm)  
m = 940  # (MeV/c^2)
b=0 #b=0,10,30 MeVfm^-3
k=100 #(MeVfm^-2)
# Define System parameters
r0 = 0
rn = 10  
N = 300  # number of points
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