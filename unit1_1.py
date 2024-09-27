import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D

# Function to plot the spherical harmonics probability density
def plot_spherical_harmonics(l, m):
    # Create a grid for the angles theta and phi
    theta = np.linspace(0, np.pi, 100)  # polar angle
    phi = np.linspace(0, 2 * np.pi, 100)  # azimuthal angle
    theta, phi = np.meshgrid(theta, phi)

    # Compute the spherical harmonics Y_l^m(theta, phi)
    Y_lm = sph_harm(m, l, phi, theta)

    # Compute the probability density |Y_l^m|^2
    probability_density = np.abs(Y_lm)**2

    # Spherical coordinates
    x = np.sin(theta) * np.cos(phi) * probability_density
    y = np.sin(theta) * np.sin(phi) * probability_density
    z = np.cos(theta) * probability_density

    # Plot the spherical harmonics probability density
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.viridis(probability_density))
    ax.set_title(f"Spherical Harmonics: l={l}, m={m}")
    
    # Hide the axes
    ax.set_axis_off()
    
    plt.show()

# Example: Plot spherical harmonics for different l and m values
plot_spherical_harmonics(0, 0)  # Example: l=0, m=0 spherical symmetry
plot_spherical_harmonics(1, 0)  # Example: l=1, m=0 or l=1, m=1 p-orbital shapes
plot_spherical_harmonics(2, 0)  # Example: l=2, m=0,1,2 d-orbital shapes

#https://images.app.goo.gl/DRJZ7GZU1m1AZTnX7 (verfication of code)