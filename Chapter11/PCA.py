import os
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

script_dir = os.path.dirname(__file__)
yelp_raw = pd.read_csv(script_dir.replace("Chapter11", "") + "data/yelp.csv")
print(yelp_raw.shape)
print(yelp_raw.head())

yelp_best_worst = yelp_raw[(yelp_raw.stars == 5) | (yelp_raw.stars == 1)]
X = yelp_best_worst.text
y = yelp_best_worst.stars == 5
print(yelp_best_worst.shape)

lr = LogisticRegression()
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=100)
vect = CountVectorizer(stop_words='english')

X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

lr.fit(X_train_dtm, y_train)
score = lr.score(X_test_dtm, y_test)
print(score)

vect = CountVectorizer(stop_words='english', max_features=100)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)
print(X_test_dtm.shape)
lr.fit(X_train_dtm, y_train)
score = lr.score(X_test_dtm, y_test)
print(score)

#crea 100 supercolonne con la PCA
from sklearn import decomposition
vect = CountVectorizer(stop_words='english')
pca = decomposition.PCA(n_components=100)
comp = 20
X_train = X_train[0:comp]
X_train_dtm = vect.fit_transform(X_train).todense()
X_train_dtm = pca.fit_transform(X_train_dtm)
X_test_dtm = vect.transform(X_test[0:comp]).todense()
X_test_dtm = pca.transform(X_test_dtm)
print(X_test_dtm.shape)
lr.fit(X_train_dtm, y_train[0:comp])
score = lr.score(X_test_dtm, y_test[0:comp])
print(score)
