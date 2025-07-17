#!/usr/bin/env python3
"""
p_star_vs_n_plot.py

Plot the equilibrium price p* as a function of the number of firms (n),
for multiple market sensitivity (alpha) values. Each line is a distinct α.

Dependencies:
    numpy
    matplotlib

Install dependencies:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Model constants
# -------------------------------------------------------------------
a = 8
b = 2
c = 1

# -------------------------------------------------------------------
# Closed-form solution for p* solving D(p, n, α) = 0
# -------------------------------------------------------------------
def p_star(alpha: float, n: int) -> float:
    """
    Compute equilibrium price p* for given alpha and n.
    Returns np.nan for invalid cases (n <= 1 or negative discriminant).
    """
    if n <= 1:
        return np.nan

    A = alpha * (n - 1)
    B = a + b * c
    C = n * b
    discriminant = (alpha**2) * ((n - 1)**2) * ((a - b * c)**2) + 4 * (n * b)**2

    if discriminant < 0:
        return np.nan

    sqrt_term = np.sqrt(discriminant)
    numerator = A * B + 2 * C - sqrt_term
    denominator = 2 * A * b

    return numerator / denominator

# -------------------------------------------------------------------
# Main plotting routine
# -------------------------------------------------------------------
def main():
    # Range of firms (n from 2 to 20)
    n_values = np.arange(2, 21)

    # Market sensitivity values
    alpha_values = [1, 2, 3, 5, 10]

    plt.figure(figsize=(10, 6))

    for alpha in alpha_values:
        # individually compute p* for each n
        p_vals = [p_star(alpha, n) for n in n_values]
        plt.plot(
            n_values,
            p_vals,
            marker='o',
            lw=2,
            label=f'α = {alpha}'
        )

    plt.title('Equilibrium Price p* vs. Number of Firms n')
    plt.xlabel('Number of Firms (n)')
    plt.ylabel('Equilibrium Price p*')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='α values')
    plt.tight_layout()
    plt.show()

main()