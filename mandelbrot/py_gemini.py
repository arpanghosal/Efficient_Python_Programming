import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(x, y, maxiter):
    c = x + 1j * y
    z = 0
    for n in range(maxiter):
        z = z**2 + c
        if abs(z) > 2:
            return n
    return maxiter

x, y = np.meshgrid(np.linspace(-2, 2, 1000), np.linspace(-2, 2, 1000))
z = np.frompyfunc(mandelbrot, 3, 1)(x, y, 100)

plt.imshow(z, cmap='viridis', interpolation='bilinear')
plt.axis('off')
plt.savefig('gem.png')
plt.show()
