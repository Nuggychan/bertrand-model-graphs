import matplotlib.pyplot as plot
import numpy

# Constants
a = 8
b = 2
c = 1

fig, axs = plot.subplots(1, 3, figsize=(24, 7), constrained_layout=True)  # 3 plots now

# --- Heatmap 1: alpha 0-10, n 0-10, both 1:1 ---
alpha_values1 = numpy.arange(0, 10.01, 0.1)  # Calculate every 0.1
n_values1 = numpy.arange(0, 10.01, 0.1)
ALPHA1, N1 = numpy.meshgrid(alpha_values1, n_values1)

# Avoid division by zero for n=1
denominator1 = 2 * ALPHA1 * (N1 - 1) * b
numerator1 = ALPHA1 * (N1 - 1) * (a + b * c) + 2 * N1 * b - numpy.sqrt((ALPHA1 * (N1 - 1) * (a - b * c))**2 + 4 * (N1 * b)**2)
p_star1 = numpy.where((N1 == 1) | (ALPHA1 == 0), numpy.nan, numerator1 / denominator1)

im1 = axs[0].imshow(
    p_star1.T,
    extent=[n_values1.min(), n_values1.max(), alpha_values1.min(), alpha_values1.max()],
    origin='lower', aspect='equal', cmap='hot'  # aspect='equal' for square shape
)
axs[0].set_title('Heatmap: alpha 0-10, n 0-10 (1:1)')
axs[0].set_xlabel('n (number of firms)')
axs[0].set_ylabel('alpha')
fig.colorbar(im1, ax=axs[0], label='p*')

# Show ticks every 1 for n, every 1 for alpha on the first heatmap
axs[0].set_xticks(numpy.arange(1, 11, 1))
axs[0].set_yticks(numpy.arange(1, 11, 1))

# --- Heatmap 2: alpha 0-100, n 0-20 ---
alpha_values2 = numpy.linspace(0, 100, 1001)  # Calculate every ~0.1
n_values2 = numpy.arange(0, 20.01, 0.1)
ALPHA2, N2 = numpy.meshgrid(alpha_values2, n_values2)

denominator2 = 2 * ALPHA2 * (N2 - 1) * b
numerator2 = ALPHA2 * (N2 - 1) * (a + b * c) + 2 * N2 * b - numpy.sqrt((ALPHA2 * (N2 - 1) * (a - b * c))**2 + 4 * (N2 * b)**2)
p_star2 = numpy.where((N2 == 1) | (ALPHA2 == 0), numpy.nan, numerator2 / denominator2)

im2 = axs[1].imshow(
    p_star2.T,
    extent=[n_values2.min(), n_values2.max(), alpha_values2.min(), alpha_values2.max()],
    origin='lower',
    aspect=0.2,  # 5:1 ratio so the sides have the same length but x/y values are preserved
    cmap='hot'  # aspect='equal' for square shape
)
axs[1].set_title('Heatmap: alpha 0-100, n 0-20')
axs[1].set_xlabel('n (number of firms)')
axs[1].set_ylabel('alpha')
fig.colorbar(im2, ax=axs[1], label='p*')

# Show ticks every 2 for n, every 10 for alpha on the second heatmap
axs[1].set_xticks(numpy.arange(2, 21, 2))
axs[1].set_yticks(numpy.arange(0, 101, 10))

# --- Heatmap 3: alpha 0-20, n 1-10 ---
alpha_values3 = numpy.arange(0, 20.01, 0.1)  # Calculate every 0.1
n_values3 = numpy.arange(1, 10.01, 0.1)
ALPHA3, N3 = numpy.meshgrid(alpha_values3, n_values3)

denominator3 = 2 * ALPHA3 * (N3 - 1) * b
numerator3 = ALPHA3 * (N3 - 1) * (a + b * c) + 2 * N3 * b - numpy.sqrt((ALPHA3 * (N3 - 1) * (a - b * c))**2 + 4 * (N3 * b)**2)
p_star3 = numpy.where((N3 == 1) | (ALPHA3 == 0), numpy.nan, numerator3 / denominator3)

im3 = axs[2].imshow(
    p_star3.T,
    extent=[n_values3.min(), n_values3.max(), alpha_values3.min(), alpha_values3.max()],
    origin='lower', aspect=0.5, cmap='hot'  # aspect=0.5 for a 2:1 width:height ratio
)
axs[2].set_title('Heatmap: alpha 0-20, n 1-10')
axs[2].set_xlabel('n (number of firms)')
axs[2].set_ylabel('alpha')
fig.colorbar(im3, ax=axs[2], label='p*')

# Show ticks every 1 for n, every 2 for alpha on the third heatmap
axs[2].set_xticks(numpy.arange(1, 11, 1))
axs[2].set_yticks(numpy.arange(2, 21, 2))

# Make the window scrollable if needed (for interactive backends)
# For matplotlib, use the built-in navigation or maximize the window.
# For Jupyter, use %matplotlib notebook or interactive widgets.
# For VS Code, you can scroll the plot pane if the figure is large.

plot.show()
