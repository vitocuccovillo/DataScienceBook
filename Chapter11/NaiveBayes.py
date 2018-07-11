import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

df = pd.read_table("https://raw.githubusercontent.com/sinanuozdemir/sfdat22/master/data/sms.tsv",
                   sep='\t', header=None, names=['label','msg'])

df.label.value_counts().plot(kind='bar')
plt.show()
print(df.label.value_counts() / df.shape[0])

df.msg = df.msg.apply(lambda x: x.lower())

spams = df[df.label == 'spam']
# probabilità che ci siano queste parole, sapendo la classe ossia spam
for w in ['send', 'cash', 'now']:
    print(w + " " + str(spams[spams.msg.str.contains(w)].shape[0] / float(spams.shape[0])))

train_simple = ['call you tonight','Call me a cab', 'please call me... PLEASE 44!']

vect = CountVectorizer()
train_simple_dtm = vect.fit_transform(train_simple)
df2 = pd.DataFrame(train_simple_dtm.toarray(), columns=vect.get_feature_names())
print(df2.head())

test_simple = ["please don't call me"]
test_simple_dtm = vect.transform(test_simple)
df3 = pd.DataFrame(test_simple_dtm.toarray(), columns=vect.get_feature_names())
print(df3.head())

X_train,X_test,y_train,y_test = train_test_split(df.msg, df.label, random_state=1)
vect = CountVectorizer()
train_dtm = vect.fit_transform(X_train) #matrice termini-doc
test_dtm = vect.transform(X_test)

#MODELLO NAIVE BAYES
nb = MultinomialNB()
nb.fit(train_dtm, y_train)
preds = nb.predict(test_dtm)

print(preds)

# confronto predizioni con etichette vere
print(metrics.accuracy_score(y_test, preds))
print(metrics.confusion_matrix(y_test, preds))