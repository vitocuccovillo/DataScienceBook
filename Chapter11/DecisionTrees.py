import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

script_dir = os.path.dirname(__file__)
titanic = pd.read_csv(script_dir.replace("Chapter11", "") + "data/titanic.csv")
titanic.Sex = titanic.Sex.map({'female': 0, 'male': 1})
titanic.Age.fillna(titanic.Age.median(), inplace=True)
emb_dummy = pd.get_dummies(titanic.Embarked, prefix="Embarked")
emb_dummy.drop(emb_dummy.columns[0], axis=1,inplace=True)

titanic = pd.concat([titanic, emb_dummy],axis=1)

feat_cols = ['Pclass', 'Sex', 'Age', 'Embarked_Q', 'Embarked_S']
X = titanic[feat_cols]
print(X.describe())
print(X.isnull().sum())
y = titanic.Survived
# addestra un albero di decisione
tree = DecisionTreeClassifier(max_depth=3, random_state=1)
tree.fit(X, y)
print(tree)
print(tree.criterion)
print(tree.n_outputs_)

imp = pd.DataFrame({'feature': feat_cols, 'importance': tree.feature_importances_})
print(imp)