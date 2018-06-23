import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt

friends = [109,59, 1190,900,65,450,31,2500,765,100]
happiness = [.6,.8,.1,.4,.2,.9,.3,.3,.6,.5]

df = pd.DataFrame({'friends':friends,'happiness':happiness})

print(df.head())

#queste due colonne hanno valori in scale diverse, con lo z-score si riportano nella stessa scala

df_scaled = pd.DataFrame(preprocessing.scale(df), columns=['friends_scaled','happiness_scaled'])
print(df_scaled.head())

#df_scaled.plot(kind='scatter',x='friends_scaled',y='happiness_scaled')
plt.scatter(x=df_scaled['friends_scaled'],y=df_scaled['happiness_scaled'])
plt.show()
