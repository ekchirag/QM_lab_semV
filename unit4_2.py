import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 940e6  # eV/c^2
hbar_c = 1973  # eV*A (hbar * c)
D = 6.755501  # eV
alpha = 1.44  # A^-1
R0 = 0.131349  # A
omega = alpha * np.sqrt(2 * D / m)

# Define r' as per problem
def r_p(r):
    return (r - R0) / r

# Morse potential function
def pot_fn(r):
    return D * (np.exp(-2 * alpha * r_p(r)) - 2 * np.exp(-alpha * r_p(r)))

# Setup discretization for radial positions
r0 = 0.131349  # Start of radial position
rn = 5  # End of radial position
N = 300  # Number of discretization points
d = (rn - r0) / (N)  # Step size in r
R = np.linspace(r0 + d, rn, N)  # Radial positions

# Initialize matrices
K = np.zeros((N, N))  # Kinetic energy matrix
V = np.zeros((N, N))  # Potential energy matrix

# Populate kinetic and potential energy matrices
for i in range(N):
    for j in range(N):
        if i == j:
            # Kinetic energy term (finite difference, central difference approximation)
            K[i, j] = -2
            # Morse potential at each radial position
            V[i, j] = pot_fn(R[i])
        elif np.abs(i - j) == 1:
            # Nearest neighbors contribute to kinetic energy matrix
            K[i, j] = 1

# Scaling the kinetic energy matrix by hbar^2 / 2m and the lattice spacing
K = -(hbar_c**2) / (2 * m * (d**2)) * K

# Construct Hamiltonian matrix H = T + V
H = K + V

# Solve the eigenvalue equation for the Hamiltonian
eVal, eVec = np.linalg.eig(H)
z = np.argsort(eVal)
energies = eVal[z]

# Print the first three energy levels in eV
print("First three energy levels (eV):", energies[:3])

# Plot the first 4 eigenstates
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(R, eVec[:, z[i]], label=f"n={i+1}", lw=1)
    plt.xlabel("r (Å)")
    plt.ylabel("Wave Function")
    plt.legend()

plt.suptitle("Wavefunctions for Schrodinger's Equation with Morse Potential")
plt.tight_layout()
plt.show()

# Plot the energy levels
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.axhline(energies[i], color='r', label=f"E_{i}" if i < 3 else "")
plt.ylabel("Energy (eV)")
plt.legend()
plt.show()
