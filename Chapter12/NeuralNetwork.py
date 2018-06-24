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

ds = ClassificationDataSet(64,1,nb_classes = 10)
test = ClassificationDataSet(64,1,nb_classes = 10)
training = ClassificationDataSet(64,1,nb_classes = 10)

for k in xrange(len(X)):
    ds.addSample(ravel(X[k]), y[k])

test_t, training_t = ds.splitWithProportion(0.25)

for k in xrange(0, test_t.getLength()):
    test.addSample(ravel(X[k]), y[k])

for k in xrange(0, training_t.getLength()):
    training.addSample(ravel(X[k]), y[k])

fnn = buildNetwork(training.indim, 64, training.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer(fnn, dataset = training, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)

trainer.trainEpochs(10)
print(percentError(trainer.testOnClassData(dataset=test), test['class']))

plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

print(fnn.activate(X[0]))

fnn = buildNetwork(training.indim, 64, training.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer(fnn, dataset= training, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)
trainer.trainEpochs(10)
print(percentError(trainer.testOnClassData(dataset=test), test['class']))
