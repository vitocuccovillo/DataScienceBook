import numpy as np
from scipy import stats
import pandas as pd
import math

np.random.seed(1234)
long_breaks = stats.poisson.rvs(loc=10, mu=60, size=3000) #3000 persone che fanno un break di 60 minuti circa
pd.Series(long_breaks).hist()

short_breaks = stats.poisson.rvs(loc=10, mu=15, size=6000) #6000 persone, con break di 15 minuti
pd.Series(short_breaks).hist()

breaks = np.concatenate((long_breaks,short_breaks))

sample_size = 100
sample = np.random.choice(a=breaks,size=sample_size)
sample_mean = sample.mean()
sample_stddev = sample.std()
sigma = sample_stddev / math.sqrt(sample_size) #stimo la dev standard della popolazione, calcolandola sul campione

interval = stats.t.interval(alpha=0.95,
                 df=sample_size-1,
                 loc=sample_mean,
                 scale=sigma)
print(interval) #la durata del break appartiene a questo intervallo (34.971995710974802, 43.628004289025192) con una
                # confidenza del 95%. Il livello di confidenza è la probabilità percentuale che l'interevallo possa contenere
                #il parametro della popolazione

