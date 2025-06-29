import numpy as np
import matplotlib.pyplot as plt
import math

# Constants
a = 8
b = 2
c = 1

def D(p, n, alpha):
    term1 = - (alpha * (n - 1) / n**2) * (a - b * p) * (p - c)
    term2 = (1 / n) * (a + b * c - 2 * b * p)
    return term1 + term2

def p_star(alpha, n):
    # Avoid division by zero for n=1
    if n == 1:
        return np.nan
    numerator = alpha * (n - 1) * (a + b * c) + 2 * n * b - math.sqrt((alpha**2) * ((n - 1)**2) * ((a - b * c)**2) + 4 * (n * b)**2)
    denominator = 2 * alpha * (n - 1) * b
    return numerator / denominator

# Values to try
alpha_values = [1, 2, 3, 5, 10]
n_values = [2, 3, 4]
p = np.linspace(0, 10, 400)

plt.figure(figsize=(15, 10))

for idx, alpha in enumerate(alpha_values, 1):
    plt.subplot(2, 3, idx)
    for n in n_values:
        D_vals = D(p, n, alpha)
        plt.plot(p, D_vals, label=f"n={n}")

        # Find x-intercepts (roots)
        sign_changes = np.where(np.diff(np.sign(D_vals)))[0]
        x_intercepts = []
        for i in sign_changes:
            # Linear interpolation for root
            p0, p1 = p[i], p[i+1]
            D0, D1 = D_vals[i], D_vals[i+1]
            if D1 != D0:
                root = p0 - D0 * (p1 - p0) / (D1 - D0)
                x_intercepts.append(root)
                plt.plot(root, 0, 'o', color=plt.gca().lines[-1].get_color())
                plt.annotate(f"{root:.2f}", (root, 0), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

        # Calculate p* for this alpha and n
        pstar = p_star(alpha, n)
        # Check if p* matches any x-intercept (within a small tolerance)
        match = any(np.isclose(root, pstar, atol=1e-2) for root in x_intercepts)
        if match:
            plt.plot(pstar, 0, 's', color='red', markersize=8, label=f"p*={pstar:.2f} (match)")
            plt.annotate("p*", (pstar, 0), textcoords="offset points", xytext=(0,-15), ha='center', color='red', fontsize=10)
        else:
            plt.plot(pstar, 0, 's', color='gray', markersize=8, label=f"p*={pstar:.2f}")

    plt.title(f"α={alpha}")
    plt.xlabel("p")
    plt.ylabel("D(p)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Payout Function D(p) for Various α and n\n(x-intercepts shown as dots, p* as squares)", fontsize=16, y=1.03)
plt.show()