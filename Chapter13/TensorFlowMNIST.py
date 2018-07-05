from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

x_mnist = mnist.train.images
y_mnist = mnist.train.labels.astype(int)
x_mnist_test = mnist.test.images
y_mnist_test = mnist.test.labels.astype(int)

plt.imshow(x_mnist[10].reshape(28,28))
print(x_mnist.shape)
print(y_mnist.shape)

feature_columns = [tf.contrib.layers.real_valued_column("",dimension=784)]
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1)
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            optimizer=optimizer,
                                            n_classes=10)
#classifier.fit(x_mnist,y_mnist, steps=1000)

print(x_mnist_test.shape)
print(y_mnist_test.shape)

#accuracy_score = classifier.evaluate(x_mnist_test, y_mnist_test)['accuracy']
#print(accuracy_score) #0.9245

### LOGREG ###

logreg = LogisticRegression()
logreg.fit(x_mnist, y_mnist)
y_predicted = logreg.predict(x_mnist_test)
accuracy = accuracy_score(y_predicted, y_mnist_test)
print("Accuratezza logreg: " + str(accuracy))


### RETE più ampia ###
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[1500],
                                            optimizer=optimizer,
                                            n_classes=10)
classifier.fit(x_mnist,y_mnist, steps=100)
accuracy_score = classifier.evaluate(x_mnist_test, y_mnist_test)['accuracy']
print(accuracy_score)