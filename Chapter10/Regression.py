import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv"
bikes = pd.read_csv(url)
print(bikes.head())

bikes.plot(kind='scatter', x='temp',y='count',alpha=0.2)
sns.lmplot(x='temp',y='count',data=bikes,aspect=1.5,scatter_kws={'alpha':0.2}) # traccia una linea di best fit (seaborn)
plt.show()

print("Correlazione: " + str(bikes[['count','temp']].corr()))

#regressione:

feature_cols = ['temp']
feature_colsOther = ['temp','season','weather','humidity']
X = bikes[feature_cols]
y = bikes['count']

linreg = LinearRegression()
linreg.fit(X,y)

#il B1 è il coefficiente di temp. Indica il modo in cui x e y si spostano insieme
# ad una variazione di 1°C, si incrementano 9 unità di noleggio
#il B0 indica indica il valore di y quando x = 0
print(linreg.coef_)
print(linreg.intercept_)
zip(feature_colsOther,linreg.coef_)
#print("Quando la temperatura è 20°, si noleggiano: " + str(linreg.predict(20)))

sns.pairplot(bikes,x_vars=feature_colsOther,y_vars='count',kind='reg')
plt.show()

#METRICHE DELLA REGRESSIONE:
y_pred = linreg.predict(X)
rmse = np.sqrt(metrics.mean_squared_error(y,y_pred))
print("errore predizione: " + str(rmse))

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X,y)
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test) # effettua le predizioni sul test set
test_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred)) #calcola l'errore sul test
print("Errore sul test set: " + str(test_error))
