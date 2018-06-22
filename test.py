import matplotlib.pyplot as plt
import numpy as np



x = np.random.randn(5)
y = np.random.randn(5)

print(x)
print(y)

plt.plot(x, y)
plt.xticks([-1, 0, 1, 2, 3, 4])
plt.show()



