import matplotlib.pyplot as plt
import numpy as np

# Generate x values
x = np.linspace(1, 2000, 1000)  # Start from 1 to avoid division by zero

# Different curve options
y1 = 1 + 1 / (1 + (500/x)**2)  # Steeper rise
y2 = 1 + 1 / (1 + np.exp(-0.01*(x-500)))  # Logistic curve

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, 'g-', linewidth=2, label='y = 1 + 1/(1 + (500/x)²)')
plt.plot(x, y2, 'orange', linewidth=2, linestyle='--', label='y = 1 + 1/(1 + e^(-0.01(x-500)))')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Elevation Profile - Steeper Rise')

plt.axhline(y=2.5, color='r', linestyle='--', alpha=0.5, label='y = 2.5 limit')
plt.legend()
plt.xlim(0, 2000)
plt.ylim(0.9, 2.6)
plt.show()