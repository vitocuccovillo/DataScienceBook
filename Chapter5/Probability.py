import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

results = []
for n in range(1,10000):
    nums = np.random.randint(low=1,high=10,size=n)
    mean = nums.mean()
    results.append(mean)

print(len(results))
df = pd.DataFrame({'mean':results})
print(df.head())
print(df.tail())

df.plot(title='Legge dei grandi numeri')
plt.xlabel("Numero di lanci nel campione")
plt.ylabel("Media del campione")
plt.show()
