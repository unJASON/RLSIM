import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
m = 100
x = np.linspace(-1, 1, m)
y_exact = 1 + 2 * x
xi = x + np.random.normal(0, 0.05, 100)
yi = 1 + 2 * xi + np.random.normal(0, 0.05, 100)
A = np.vstack([xi**0, xi**1])
sol, r, rank, s = la.lstsq(A.T, yi)   #求取各个系数大小
y_fit = sol[0] + sol[1] * x

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(xi, yi, 'go', alpha=0.5, label='Simulated data')
ax.plot(x, y_exact, 'k', lw=2, label='True value y = 1 + 2x')
ax.plot(x, y_fit, 'b', lw=2, label='Least square fit')
ax.set_xlabel("x", fontsize=18)
ax.set_ylabel("y", fontsize=18)
ax.legend(loc=2)         #设置曲线标注位置
plt.show()