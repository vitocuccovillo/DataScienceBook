import pandas as pd
from matplotlib import pyplot as plt
import os

hoursTV = [0,0,0,1,1.3,1.4,2,2.1,2.6,3.2,4.1,4.4,4.4,5]
workPerf = [87,89,92,90,82,80,77,80,76,85,80,75,73,72]

df = pd.DataFrame({'HoursTV': hoursTV, 'workPerformance': workPerf})
print(df.head())
plt.scatter(x=df['HoursTV'], y=df['workPerformance'])
plt.show()

print("Correlazione: " + str(df.corr()))

# bar chart
script_dir = os.path.dirname(__file__)
drinks = pd.read_csv(script_dir.replace("Chapter9","") + "data/drinks.csv")
drinks.continent.value_counts().plot(kind='bar', title='Nazioni per Continente')
plt.xlabel("Continente")
plt.ylabel("Count")
plt.show()

drinks.groupby('continent').beer_servings.mean().plot(kind='bar')
plt.show()

#boxplot
drinks.boxplot(column='beer_servings',by='continent')
plt.show()