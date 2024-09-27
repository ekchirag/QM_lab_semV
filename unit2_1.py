import numpy as np
import matplotlib.pyplot as plt

# Define Constant parameters
h_cut = 1973.0  # (eV*A)  
m = 0.511e6  # (eV/c^2)

# Define System parameters
x0 = 0.00 
xn = 10.0  
N = 150  # number of points
d = (xn - x0) / (N+1)  # Step Size
X = np.linspace(x0, xn, N)  #  positions in x direction

K = np.zeros((len(X), len(X)))  # Kinetic energy matrix

for i in range(len(X)):
    for j in range(len(X)):
        if i == j:
            K[i,j] = -2
        elif np.abs(i - j) == 1:
            K[i,j] = 1

# Scaling the kinetic energy matrix
K = (-(h_cut**2) / (2 * m * (d**2))) * K

# Constructing the Hamiltonian matrix
H = K 

# Solve the eigenvalue equation
eVal, eVec = np.linalg.eig(H)
z = np.argsort(eVal)
energies = eVal[z]

# Print the first three energy levels in eV
print("First three energy levels (eV):", energies[:3])


#plot the first 4 eigen states 
color=["r",'b','g','b']
plt.figure(figsize=(10,6))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(X, eVec[:,z[i]],label=f"n={i+1}", lw=1,color=color[i])
    plt.axhline(0, color='k' , lw=0.4)
    plt.xlabel("x (Å)")
    plt.ylabel("Wave Function")
    plt.legend()
plt.suptitle("Particle in a box Equation")

plt.figure(figsize=(10,6))
for i in range(6):
    k=i%4
    plt.axhline(energies[i], color=color[k],label=f"E{i+1}= {energies[i]}",lw=0.4)
    plt.legend()
plt.suptitle("Energy levels of the particle in a box")
plt.ylabel("Energy values")
plt.legend()
plt.show()