import pandas as pd
import os

script_dir = os.path.dirname(__file__)
titanic = pd.read_csv(script_dir.replace("Chapter6","") + "data/titanic.csv")
titanic = titanic[['Sex','Survived']]
print(titanic.head())

numRows = float(titanic.shape[0])
p_survived = (titanic.Survived == 1).sum() / numRows
p_notsurvived = 1 - p_survived

print("Probabilità di sopravvivenza: " + str(p_survived))
print("Probabilità di non sopravvivenza: " + str(p_notsurvived))

p_male = (titanic.Sex == "male").sum() / numRows
p_female = 1 - p_male

print("Probabilità uomo: " + str(p_male))
print("Probabilità donna: " + str(p_female))

num_women = titanic[titanic.Sex == "female"].shape[0]
women_lived = titanic[(titanic.Sex == "female") & (titanic.Survived == 1)].shape[0]
p_survived_given_women = women_lived / float(num_women)
print("Probabilità di sopravvivenza delle donne: " + str(p_survived_given_women))
