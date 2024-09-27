import numpy as np
import matplotlib.pyplot as plt


def pot_fn(x):
    return (m * (w**2) * (x**2)) * (1 / 2)


# Define Constant parameters
h_cut = 1973.0  # (eV*A)
m = 0.511e6  # (eV/c^2)
w = 10e-3 
print(w)
# Define System parameters
x0 = -10.00
xn = 10.0
N = 500  # number of points
d = (xn - x0) / (N)  # Step Size
X = np.linspace(x0 , xn , N)  #  positions

K = np.zeros((len(X), len(X)))  # Kinetic energy matrix
V = np.zeros((len(X), len(X)))  # Potential energy matrix

for i in range(len(X)):
    for j in range(len(X)):
        if i == j:
            K[i, j] = -2
            V[i, j] = pot_fn(X[i])
        elif np.abs(i - j) == 1:
            K[i, j] = 1

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
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(X, eVec[:, z[i]], label=f"n={i+1}, E{i+1}={energies[i]}", lw=1)
    plt.axhline(0,color='k',lw=.3)
    plt.xlabel("x (Å)")
    plt.ylabel("Wave Function")
    plt.legend()
plt.suptitle("Solution of 1-D Finite Potential Well")

plt.figure(figsize=(10, 6))
for i in range(10):
    plt.axhline(energies[i], color='r')
plt.ylabel("Energy values")
plt.legend()

plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)
plt.plot(X, eVec[:, z[100]], lw=.5)
plt.axhline(0,color='k',lw=.3)
plt.title(f"Wave Function at E = {energies[100]}")
plt.subplot(1,2,2)
plt.plot(X, eVec[:, z[100]]**2, lw=.6)
plt.axhline(0,color='k',lw=.3)
plt.title(f"Probability density at E = {energies[100]}")
plt.suptitle("100th excited state")
plt.show()