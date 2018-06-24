import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

X = pd.read_table("https://raw.githubusercontent.com/sinanuozdemir/sfdat22/master/data/sms.tsv",
                   sep='\t',header=None,names=['label','msg'])

d = 1
print(X.shape[0])
print(X.head())

vect = CountVectorizer()
train = vect.fit_transform(X)
df = pd.DataFrame(train.toarray(),columns=vect.get_feature_names())
df.msg = df.msg.apply(lambda x:x.lower())

print(df.head())
X_first_word = df[:,:1]
neigh = NearestNeighbors(n_neighbors=X.shape[0])
neigh.fit(X_first_word)
A = neigh.kneighbors_graph(X_first_word, mode='distance').todense()
num_points_within_d = (A < d).sum()

print(num_points_within_d)