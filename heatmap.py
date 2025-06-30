import matplotlib.pyplot as plot
import numpy

# Constants
a = 8
b = 2
c = 1

# Ranges for alpha and n
alpha_values = numpy.linspace(0.1, 10, 100)  # avoid alpha=0 to prevent division by zero
n_values = numpy.arange(2, 21)

# Create meshgrid
ALPHA, N = numpy.meshgrid(alpha_values, n_values)

# Compute p*
numerator = ALPHA * (N - 1) * (a + b * c) + 2 * N * b - numpy.sqrt((ALPHA * (N - 1) * (a - b * c))**2 + 4 * (N * b)**2)
denominator = 2 * ALPHA * (N - 1) * b
p_star = numerator / denominator

# Plot heatmap with switched axes
plot.imshow(p_star.T, extent=[n_values.min(), n_values.max(), alpha_values.min(), alpha_values.max()],
            origin='lower', aspect='auto', cmap='hot')
plot.colorbar(label='p*')
plot.xlabel('n (number of firms)')
plot.ylabel('alpha')
plot.title('Heatmap of p* as a function of n and alpha')

# Set alpha ticks every 1 unit
plot.yticks(numpy.arange(1, 11, 1))

plot.show()
