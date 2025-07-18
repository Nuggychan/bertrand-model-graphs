import numpy as np
import matplotlib.pyplot as plt

# Constants
a = 8
b = 2
c = 1

def p_star(alpha, n):
    """
    Closed-form equilibrium price p* solving D(p, n, α) = 0.
    Returns np.nan for n=1 (degenerate case).
    """
    if n == 1:
        return np.nan
    A = alpha * (n - 1)
    B = a + b * c
    C = n * b
    D_sqrt = np.sqrt(
        (alpha**2) * ((n - 1)**2) * ((a - b * c)**2)
        + 4 * (n * b)**2
    )
    numerator = A * B + 2 * C - D_sqrt
    denominator = 2 * A * b
    return numerator / denominator

# Range of firms
n_values = np.arange(2, 21)    # e.g. from 2 up to 20 firms

# Different α values to compare
alpha_values = [1, 2, 3, 5, 10, 20, 30]

plt.figure(figsize=(10, 6))

for alpha in alpha_values:
    # compute p* for each n
    p_stars = [p_star(alpha, n) for n in n_values]
    plt.plot(
        n_values,
        p_stars,
        marker='o',
        linewidth=2,
        label=f'α = {alpha}'
    )

plt.xlabel('Number of Firms (n)')
plt.ylabel('Equilibrium Price p*')
plt.title('p* vs. Number of Firms for Different α Values')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='α Values')
plt.tight_layout()
plt.show()