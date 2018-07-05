from sklearn import datasets, metrics
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

### TENSORFLOW ###
# le feature hanno valori reali, ho 4 colonne, dunque la dimensione è 4
feature_columns = [tf.contrib.layers.real_valued_column("",dimension=4)]

# learning rate, ossia velocità di apprendimento
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# costruisce un classificatore lineare (regressione logistica).
# dico a Tensorflow il numero diclassi, 3
classifier = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns, optimizer=optimizer, n_classes=3)

# adattare il modello tramite l'ottimizzazione dell'errore con la discesa a gradiente stocastico
classifier.fit(x=X_train, y=y_train,steps=1000) #iterazioni
# se il modello dovesse apprendere troppo velocemente potrebbe mancare la risposta

accuracy = classifier.evaluate(x=X_test, y=y_test)['accuracy']
print("accuracy: " + str(accuracy))

# classifica nuovi fiori
new_samples = np.array([[6.4,3.2,4.5,1.5],[5.8,3.1,5.0,1.7]], dtype=float)
y = classifier.predict(new_samples)
print(str(y))

### LOGISTIC REGRESSION ###
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_predicted = logreg.predict(X_test) # testa sull'insieme di collaudo
accuracy = metrics.accuracy_score(y_predicted, y_test)
print("LOGREG accuracy: " + str(accuracy))

### AUMENTARE LE ITERAZIONI NELLA NN ###
classifier.fit(x=X_train, y=y_train,steps=2000) #iterazioni
accuracy = classifier.evaluate(x=X_test, y=y_test)['accuracy']
print("accuracy 2000 iter: " + str(accuracy))

### DEEP NEURAL NETWORK ###
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            optimizer=optimizer,
                                            n_classes=3)

classifier.fit(x=X_train,y=y_train, steps=3000)

accuracy_score = classifier.evaluate(x=X_test, y=y_test)['accuracy']
print("Accuracy DNN: " + str(accuracy_score))
