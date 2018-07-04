import statsmodels.api as sm
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
import pandas as pd

affairs_df = sm.datasets.fair.load_pandas().data
print(affairs_df.head())

# creo un nuovo campo nel dataframe, variabile categorica
affairs_df['affair_binary'] = (affairs_df['affairs'] > 0)

sb.heatmap(affairs_df.corr())
plt.show()

affairs_X = affairs_df.drop(['affairs', 'affair_binary'], axis= 1)
affairs_y = affairs_df['affair_binary']

model = DecisionTreeClassifier()
scores = cross_val_score(model, affairs_X, affairs_y, cv= 10)

print("Accuracy: " + str(scores.mean()))
print("Variance: " + str(scores.std()))

model.fit(affairs_X,affairs_y)
#stampo le feature in base all'importanza
print(pd.DataFrame({'feature': affairs_X.columns, 'importance': model.feature_importances_}))

# variabili dummy
# le variabili dummy servono per codificare variabili qualitative usando colonne distinte
occupation_dummy = pd.get_dummies(affairs_df['occupation'], prefix='occ_').iloc[:,1:]
affairs_df = pd.concat([affairs_df, occupation_dummy], axis=1)
print(affairs_df.head())
occupation_dummies = pd.get_dummies(affairs_df['occupation_husb'], prefix='occ_husb').iloc[:,1:]
affairs_df = pd.concat([affairs_df, occupation_dummies], axis=1)
print(affairs_df.head())

affairs_X = affairs_df.drop(['affairs', 'affair_binary', 'occupation', 'occupation_husb'], axis=1)
affairs_y = affairs_df['affair_binary']

model = DecisionTreeClassifier()
scores = cross_val_score(model, affairs_X, affairs_y, cv=10)
print("Accuracy:" + str(scores.mean()))
print("Variance:" + str(scores.std()))

model.fit(affairs_X, affairs_y)
print(pd.DataFrame({'feature': affairs_X.columns, 'importance': model.feature_importances_})
      .sort_values(by='importance'))

#ordina il dataframe con sort_values(by='campo')