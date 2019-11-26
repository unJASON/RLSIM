import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-1, 1, 100)
a, b, c = 1, 2, 3
y_exact = a + b * x + c * x**2
m = 100
xi=1 - 2 * np.random.rand(m)
yi=a + b * xi + c * xi**2 + np.random.randn(m)
A = np.vstack([xi**0, xi**1, xi**2])
sol, r, rank, s = la.lstsq(A.T, yi)
y_fit = sol[0] + sol[1] * x + sol[2] * x**2
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(xi, yi, 'go', alpha=0.5, label='Simulated data')
ax.plot(x, y_exact, 'k', lw=2, label='True value $y = 1 + 2x + 3x^2$')
ax.plot(x, y_fit, 'b', lw=2, label='Least square fit')
ax.set_xlabel("x", fontsize=18)
ax.set_ylabel("y", fontsize=18)
ax.legend(loc=2)
plt.show()