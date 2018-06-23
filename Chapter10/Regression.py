import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv"
bikes = pd.read_csv(url)
print(bikes.head())

bikes.plot(kind='scatter', x='temp',y='count',alpha=0.2)
sns.lmplot(x='temp',y='count',data=bikes,aspect=1.5,scatter_kws={'alpha':0.2})
plt.show()