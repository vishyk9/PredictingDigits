
# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import svm



digits = datasets.load_digits()
clf = svm.SVC(gamma = .001, C = 100)



x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

print('Prediction:', clf.predict(digits.data[[-1]]))
plt.imshow(digits.images[-1])
plt.show()
