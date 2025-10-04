import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import data

# Step 1: Load a sample image (grayscale)
image = data.coins()  # Using 'coins' as a sample image
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Step 2: Compute the Gray Level Co-occurrence Matrix (GLCM)
# Define the offsets for GLCM calculation (e.g., horizontal, vertical, diagonal)
distances = [1,2,5]  # distance of 1 pixel
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # angles: 0, 45, 90, 135 degrees

# Compute the GLCM
glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

# Step 3: Compute properties of the GLCM (e.g., contrast, correlation, energy, homogeneity)
contrast = graycoprops(glcm, prop='contrast')
correlation = graycoprops(glcm, prop='correlation')
energy = graycoprops(glcm, prop='energy')
homogeneity = graycoprops(glcm, prop='homogeneity')

# Display the properties
print("GLCM Contrast:\n", contrast)
print("GLCM Correlation:\n", correlation)
print("GLCM Energy:\n", energy)
print("GLCM Homogeneity:\n", homogeneity)

# Step 4: Apply normalization on the image (simple min-max normalization)
normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))

# Plot the normalized image
plt.imshow(normalized_image, cmap='gray')
plt.title('Normalized Image')
plt.axis('off')
plt.show()

# Step 5: Plot sine and cosine kernels
x = np.linspace(-10, 10, 100)
sine_kernel = np.sin(x)
cosine_kernel = np.cos(x)

plt.figure(figsize=(10, 5))

# Plot sine kernel
plt.subplot(1, 2, 1)
plt.plot(x, sine_kernel, label='Sine Kernel', color='b')
plt.title('Sine Kernel')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)

# Plot cosine kernel
plt.subplot(1, 2, 2)
plt.plot(x, cosine_kernel, label='Cosine Kernel', color='r')
plt.title('Cosine Kernel')
plt.xlabel('x')
plt.ylabel('cos(x)')
plt.grid(True)

plt.tight_layout()
plt.show()
