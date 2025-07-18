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
    A = alpha * (n - 1)
    B = a + b * c
    C = n * b
    D_sqrt = np.sqrt((alpha**2) * ((n - 1)**2) * ((a - b * c)**2) + 4 * (n * b)**2)
    numerator = A * B + 2 * C - D_sqrt
    denominator = 2 * A * b
    return numerator / denominator

def p_conjugate(alpha, n):
    # Avoid division by zero for n=1
    if n == 1:
        return np.nan
    A = alpha * (n - 1)
    B = a + b * c
    C = n * b
    D_sqrt = np.sqrt((alpha**2) * ((n - 1)**2) * ((a - b * c)**2) + 4 * (n * b)**2)
    numerator = A * B + 2 * C + D_sqrt  # Only sign change here
    denominator = 2 * A * b
    return numerator / denominator

# Values to try
alpha_values = [1, 2, 3, 5, 10]
n_values = [2, 3, 4]
p = np.linspace(0, 10, 400)

plt.figure(figsize=(12, 12), constrained_layout=True)  # Reduced figure size

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

        # Calculate p* and p^ for this alpha and n
        pstar = p_star(alpha, n)
        pconj = p_conjugate(alpha, n)

        # Check if p* matches any x-intercept (within a small tolerance)
        match_star = any(np.isclose(root, pstar, atol=1e-2) for root in x_intercepts)
        if match_star:
            plt.plot(pstar, 0, 's', color='red', markersize=8, label=f"p*={pstar:.2f} (match)")
            plt.annotate("p*", (pstar, 0), textcoords="offset points", xytext=(0,-15), ha='center', color='red', fontsize=10)
        else:
            plt.plot(pstar, 0, 's', color='gray', markersize=8, label=f"p*={pstar:.2f}")

        # Check if p^ matches any x-intercept (within a small tolerance)
        match_conj = any(np.isclose(root, pconj, atol=1e-2) for root in x_intercepts)
        if match_conj:
            plt.plot(pconj, 0, 'D', color='blue', markersize=8, label=f"p^={pconj:.2f} (match)")
            plt.annotate("p^", (pconj, 0), textcoords="offset points", xytext=(0,-25), ha='center', color='blue', fontsize=10)
        else:
            plt.plot(pconj, 0, 'D', color='cyan', markersize=8, label=f"p^={pconj:.2f}")

    plt.title(f"α={alpha}")
    plt.xlabel("Strategy Price p")
    plt.ylabel("Selection Gradient D(p)")
    plt.ylim(-10, 50)  # Set y-axis scale for all plots
    plt.legend(fontsize=8)  # Make the legends slightly smaller
    plt.grid(True)

# Add a final graph with all equations
plt.subplot(2, 3, 6)
for alpha in alpha_values:
    for n in n_values:
        D_vals = D(p, n, alpha)
        plt.plot(p, D_vals, label=f"α={alpha}, n={n}")
plt.title("All Equations")
plt.xlabel("p")
plt.ylabel("D(p)")
plt.ylim(-10, 50)
plt.legend(fontsize=6, ncol=2)
plt.grid(True)

plt.show()