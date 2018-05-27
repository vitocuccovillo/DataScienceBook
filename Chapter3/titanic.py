import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    titanic = pd.read_csv(script_dir.replace("Chapter3","") + "data/titanic.csv")
    print(titanic.head())
    print(titanic.shape)

    titanic['Sex'] = np.where(titanic['Sex'] == 'female',1,0) # sostituisco la colonna sex con dei valori boolean

    print(titanic.describe())

    print("Valori nulli in age:" + str(sum(titanic['Age'].isnull())))
    print(titanic.isnull().sum()) # conta i nulli di ogni colonna e li somma

    age_avg = titanic['Age'].mean()
    titanic['Age'].fillna(age_avg, inplace=True) #sostituisco i nulli della colonna age con la media
    print(titanic.groupby('Sex')['Age'].mean()) #raggruppa i dati, 1-vuol dire donna, e di ogni gruppo calcola la media dell'età
    titanic['Sex'].value_counts().plot(kind='bar')
    plt.show()