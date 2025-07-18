
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
a = 8
b = 2
c = 1


# Compute equilibrium price p* for given α and n
def p_star(alpha: float, n: int) -> float:
    if n <= 1:
        return np.nan

    A = alpha * (n - 1)
    B = a + b * c
    C = n * b

    discriminant = (alpha**2) * ((n - 1)**2) * ((a - b * c)**2) + 4 * (C**2)
    if discriminant < 0:
        return np.nan

    sqrt_D = np.sqrt(discriminant)
    numerator = A * B + 2 * C - sqrt_D
    denominator = 2 * A * b
    return numerator / denominator


# Compute equilibrium payoff π(p*) = (1/n)(a - bp*)(p* - c)
def payoff(alpha: float, n: int) -> float:
    p = p_star(alpha, n)
    if np.isnan(p):
        return np.nan
    return (1 / n) * (a - b * p) * (p - c)

def main():
    n_values = np.arange(2, 21)              # Firms from 2 to 20
    alpha_values = [1, 2, 3, 5, 10, 20, 30]          # Market sensitivity levels

    plt.figure(figsize=(10, 6))

    for alpha in alpha_values:
        payoff_vals = [payoff(alpha, n) for n in n_values]
        plt.plot(n_values, payoff_vals, marker='o', linewidth=2, label=f'α = {alpha}')

    plt.title("Firm's Payoff π(p*) vs. Number of Firms n")
    plt.xlabel("Number of Firms (n)")
    plt.ylabel("Equilibrium Payoff π(p*)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='α values')
    plt.tight_layout()
    plt.show()

main()