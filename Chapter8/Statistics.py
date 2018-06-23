import numpy as np
from scipy import stats as stats
import pandas as pd
from matplotlib import pyplot as plt
import random

np.random.seed(1234)
long_breaks = stats.poisson.rvs(loc=10, mu=60, size=3000) #3000 persone che fanno un break di 60 minuti circa
pd.Series(long_breaks).hist()

short_breaks = stats.poisson.rvs(loc=10, mu=15, size=6000) #6000 persone, con break di 15 minuti
pd.Series(short_breaks).hist()

breaks = np.concatenate((long_breaks,short_breaks))
plt.hist(breaks)
plt.show()

print("Pausa media: " + str(breaks.mean()))

sample_breaks = np.random.choice(a=breaks,size=100) #campione di 100 dipend
print(breaks.mean() - sample_breaks.mean())

employee_races = (["white"]*2000) + (["black"]*1000) + (["hispanic"]*1000) + (["asian"]*3000) + (["other"]*3000)
demo_sample = random.sample(employee_races, 1000)

for race in set(demo_sample):
    print(race + " proporz. stimata: " + str(demo_sample.count(race)/1000))

plt.hist(breaks, range=(5,100), bins=50) #dati bimodali, si hanno due campane distinte
plt.show()

point_estimates = []
for x in range(500): #genera 500 campioni
    sample = np.random.choice(a=breaks,size=100) # ne prende 100
    point_estimates.append(sample.mean())

plt.hist(point_estimates)
plt.show() #distribuzione delle medie del campione