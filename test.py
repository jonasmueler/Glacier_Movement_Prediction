import numpy as np
import matplotlib.pyplot as plt

# Generate x values from 0 to 2*pi
x = np.linspace(1, 30, 1000)
y = np.linspace(1, 12, 1000)

# Calculate y values for sine function
y = np.sin((2*x + 2*y)/100)

# Plot the function
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Function')
plt.show()