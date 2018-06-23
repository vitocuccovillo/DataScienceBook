from scipy import stats
import numpy as np

long_breaks_eng = stats.poisson.rvs(loc=10, mu=55, size=100)
short_breaks_eng = stats.poisson.rvs(loc=10, mu=55, size=300)

long_breaks = stats.poisson.rvs(loc=10, mu=60, size=3000)
short_breaks = stats.poisson.rvs(loc=10, mu=15, size=6000)

breaks = np.concatenate((long_breaks,short_breaks))
breaks_eng = np.concatenate((long_breaks_eng,short_breaks_eng))

print(breaks.mean())
print(breaks_eng.mean())

t_statistic, p_value = stats.ttest_1samp(a=breaks_eng, popmean=breaks.mean())

print("t_statistic: " + str(t_statistic))
print("p_value: " + str(p_value))
