#code 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants
hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
m = 9.10938356e-31    # Mass of electron (kg)
e = 1.60218e-19       # Elementary charge (Coulombs or Joules per eV)
n=14                   #number of states
# Define transcendental equations for even and odd solutions
def transcendental_even(E, a, V0):
    if E <= 0 or E >= V0:
        return np.inf  # Invalid domain for E
    k = np.sqrt(2 * m * E) / hbar
    kappa = np.sqrt(2 * m * (V0 - E)) / hbar
    return k * np.tan(k * a) - kappa

def transcendental_odd(E, a, V0):
    if E <= 0 or E >= V0:
        return np.inf  # Invalid domain for E
    k = np.sqrt(2 * m * E) / hbar
    kappa = np.sqrt(2 * m * (V0 - E)) / hbar
    return k/np.tan(k * a) + kappa

# Solve for energy eigenvalues
def find_eigenvalues(a, V0, num_states=n):
    E_guess = np.linspace(1e-20, V0 - 1e-20, 1000)  # Ensure that E < V0
    eigenvalues = []
    eigenvalues_even=[]
    eigenvalues_odd=[]
    # Solve for even solutions
    for E in E_guess:
        try:
            E_even = fsolve(transcendental_even, E, args=(a, V0), xtol=1e-10)[0]
            if 0 < E_even < V0 and E_even not in eigenvalues:
                eigenvalues.append(E_even)
                eigenvalues_even.append(E_even)
        except:
            pass
    
    # Solve for odd solutions
    for E in E_guess:
        try:
            E_odd = fsolve(transcendental_odd, E, args=(a, V0), xtol=1e-10)[0]
            if 0 < E_odd < V0 and E_odd not in eigenvalues:
                eigenvalues.append(E_odd)
                eigenvalues_odd.append(E_odd)
        except:
            pass
    print('eigenvalues_odd :',sorted(eigenvalues_odd)[:num_states])
    print('eigenvalues_even :',sorted(eigenvalues_even)[:num_states])
    return sorted(eigenvalues)[:num_states]


# Define the wavefunction inside and outside the well
def wavefunction(x, E, a, V0):
    if E <= 0 or E >= V0:
        return 0  # Invalid energy
    
    k = np.sqrt(2 * m * E) / hbar
    kappa = np.sqrt(2 * m * (V0 - E)) / hbar
    
    if np.abs(x) <= a:
        return np.cos(k * x) if np.isclose(k, 0) else np.sin(k * x)
    else:
        return np.exp(-kappa * (np.abs(x) - a))

# Normalize the wavefunction
def normalize_wavefunction(x, psi):
    norm_factor = np.sqrt(np.trapz(np.abs(psi)**2, x))
    return psi / norm_factor

# Plot the wavefunctions for the first few eigenstates
def plot_eigenfunctions(a, V0, eigenvalues):
    x = np.linspace(-2*a, 2*a, 1000)
    
    plt.figure(figsize=(10, 6))
    
    for n, E in enumerate(eigenvalues):
        psi = np.array([wavefunction(xi, E, a, V0) for xi in x])
        psi_normalized = normalize_wavefunction(x, psi)
        plt.plot(x, psi_normalized, label=f'State {n}, E={E/e:.2f} eV')
    
    plt.xlabel('x (meters)')
    plt.ylabel('Wavefunction ψ(x)')
    plt.title('Normalized Eigenfunctions in a Finite Potential Well')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters for the potential well
a = 1e-10  # Width of the well (meters)
V0 = 50 * e  # Depth of the well (50 eV in Joules)
# Find the eigenvalues and plot the corresponding eigenfunctions
eigenvalues = find_eigenvalues(a, V0, num_states=n)
plot_eigenfunctions(a, V0, eigenvalues)


'''
# code 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants
hbar = 1.0  # Planck's constant (in units such that hbar = 1)
m = 1.0     # Mass of particle (in units such that m = 1)
a = 1.0     # Half-width of the potential well
V0 = 5.0    # Depth of the potential well

# Define the transcendental equations

def even_transcendental(k, V0, a):
    """Transcendental equation for even wavefunctions"""
    return k * np.tan(k * a) - np.sqrt(2 * m * V0 - k**2)

def odd_transcendental(k, V0, a):
    """Transcendental equation for odd wavefunctions"""
    return -k * np.tan(k * a) - np.sqrt(2 * m * V0 - k**2)

# Numerically solve for k values (roots of the transcendental equations)
def find_eigenvalues(V0, a):
    """Find the eigenvalues by solving the transcendental equations"""
    # Guess range for k values
    k_guesses = np.linspace(0.1, np.sqrt(2 * m * V0), 100)
    
    # Store the solutions for even and odd wavefunctions
    even_k_vals = []
    odd_k_vals = []

    # Solve for even wavefunctions
    for guess in k_guesses:
        root, = fsolve(even_transcendental, guess, args=(V0, a))
        if root > 0 and root not in even_k_vals:
            even_k_vals.append(root)

    # Solve for odd wavefunctions
    for guess in k_guesses:
        root, = fsolve(odd_transcendental, guess, args=(V0, a))
        if root > 0 and root not in odd_k_vals:
            odd_k_vals.append(root)
    
    return np.array(even_k_vals), np.array(odd_k_vals)

# Calculate energy eigenvalues from k values
def energy_eigenvalues(k_vals):
    """Calculate energy eigenvalues E = hbar^2 k^2 / 2m"""
    return (hbar**2 * k_vals**2) / (2 * m)

# Eigenfunctions
def eigenfunction_even(x, k, a):
    """Even eigenfunction inside the well"""
    A = np.sqrt(2 / a)  # Normalization constant for even function
    return A * np.cos(k * x)

def eigenfunction_odd(x, k, a):
    """Odd eigenfunction inside the well"""
    A = np.sqrt(2 / a)  # Normalization constant for odd function
    return A * np.sin(k * x)

# Plot eigenfunctions
def plot_eigenfunctions(even_k_vals, odd_k_vals, a):
    """Plot the normalized eigenfunctions"""
    x_vals = np.linspace(-2*a, 2*a, 1000)
    
    # Plot even eigenfunctions
    plt.figure(figsize=(10, 6))
    for i, k in enumerate(even_k_vals):
        psi_even = eigenfunction_even(x_vals, k, a)
        plt.plot(x_vals, psi_even, label=f'Even n={i+1}')
    
    # Plot odd eigenfunctions
    for i, k in enumerate(odd_k_vals):
        psi_odd = eigenfunction_odd(x_vals, k, a)
        plt.plot(x_vals, psi_odd, label=f'Odd n={i+1}')
    
    plt.title("Normalized Eigenfunctions for Finite Potential Well")
    plt.xlabel("Position x")
    plt.ylabel("Wavefunction ψ(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
even_k_vals, odd_k_vals = find_eigenvalues(V0, a)
even_eigenvalues = energy_eigenvalues(even_k_vals)
odd_eigenvalues = energy_eigenvalues(odd_k_vals)

print("Even eigenvalues:", even_eigenvalues)
print("Odd eigenvalues:", odd_eigenvalues)

# Plot eigenfunctions
plot_eigenfunctions(even_k_vals, odd_k_vals, a)
'''