import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split


def when_is_it(hour):

    if hour >= 5 and hour < 11:
        return 'morning'
    if hour >= 11 and hour < 16:
        return 'afternoon'
    if hour >= 16 and hour < 18:
        return 'rush_hours'
    else:
        return 'off_hours'


url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/bikeshare.csv"
bikes = pd.read_csv(url)
print(bikes.head())

avg_bikes_rental = bikes['count'].mean()
bikes['above_avg'] = bikes['count'] >= avg_bikes_rental

feature_cols = ['temp']
X=bikes[feature_cols]
y=bikes['above_avg']

X_train,X_test,y_train,y_test = train_test_split(X,y)

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

score = logreg.score(X_test, y_test)
print(score)

#definizione di una nuova colonna categorica a partire dalla data

bikes['hour'] = bikes['datetime'].apply(lambda x:int(x[11]+x[12]))
print(bikes['hour'].tail())

bikes['when_is_it'] = bikes['hour'].apply(when_is_it)
print(bikes[['when_is_it','above_avg']].head())

bikes.groupby('when_is_it').above_avg.mean().plot(kind='bar')
plt.show()

#nuove colonne per la regressione logistica
when_dummies = pd.get_dummies(bikes['when_is_it'],prefix='when__')
when_dummies = when_dummies.iloc[:,1:]

X = when_dummies
y = bikes.above_avg
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
score = logreg.score(X_test, y_test)
print(score)

new_bike = pd.concat((bikes[['temp','humidity']],when_dummies), axis=1)
X = new_bike
y = bikes.above_avg
X_train,X_test,y_train,y_test = train_test_split(X,y)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
score = logreg.score(X_test, y_test)
print(score)