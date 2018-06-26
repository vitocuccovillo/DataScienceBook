#from pybrain.datasets import ClassificationDataSet
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules.softmax import SoftmaxLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from numpy import ravel
from scipy._lib.six import xrange
from sklearn import datasets
from matplotlib import pyplot as plt
import sys
#from pybrain.datasets import ClassificationDataSet

digits = datasets.load_digits()
X, y = digits.data, digits.target

print(X[0].shape)

ds = ClassificationDataSet(64, 10, nb_classes = 10)
test = ClassificationDataSet(64, 10, nb_classes = 10)
training = ClassificationDataSet(64, 10, nb_classes = 10)

for k in xrange(len(X)):
    ds.addSample(ravel(X[k]), y[k])

test_t, training_t = ds.splitWithProportion(0.25)

for k in xrange(0, test_t.getLength()):
    test.addSample(test_t.getSample(k)[0], test_t.getSample(k)[1])

for k in xrange(0, training_t.getLength()):
    training.addSample(training_t.getSample(k)[0], training_t.getSample(k)[1])

print(training.getLength())
print(test.getLength())

print(test.indim)
print(test.outdim)
print(training.indim)
print(training.outdim)

fnn = buildNetwork(training.indim, 64, training.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer(fnn, dataset = training, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)
trainer.trainEpochs(10)

print(percentError(trainer.testOnClassData(), training['class']))
print(percentError(trainer.testOnClassData(dataset=test), test['class']))

plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

for i in range(0,10):
    print(fnn.activate(X[i]))
