import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gradient_size = 5
gradient = np.zeros((gradient_size, gradient_size, 3), dtype=np.uint8)
for x in range(gradient.shape[0]):
    for y in range(gradient.shape[1]):
        r = 255
        gb = (255 * ((5 * x + y) / 24))
        gradient[y,x] = (r, gb, gb)

print(gradient[-1])
plt.imshow(gradient)
#plt.savefig('gradient.png')