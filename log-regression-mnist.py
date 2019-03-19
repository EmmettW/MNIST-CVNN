# Logistic Regression for MNIST
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression

mnist = fetch_mldata('MNIST original')
X,y = mnist["data"], mnist["target"]

shuffle_index = np.random.permutation(X_12.shape[0])
X_12_shuffled, y_12_shuffled = X_12[shuffle_index], y_12[shuffle_index]

train_proportion = 0.8
train_test_cut = int(len(X_12)*train_proportion)

X_train, X_test, y_train, y_test = \
    X_12_shuffled[:train_test_cut], \
    X_12_shuffled[train_test_cut:], \
    y_12_shuffled[:train_test_cut], \
    y_12_shuffled[train_test_cut:]
    
print("Shape of X_train is", X_train.shape)
print("Shape of X_test is", X_test.shape)
print("Shape of y_train is", y_train.shape)
print("Shape of y_test is", y_test.shape)

X_train_normalised = X_train/255.0
X_test_normalised = X_test/255.0

X_train_tr = X_train_normalised.transpose()
y_train_tr = y_train.reshape(1,y_train.shape[0])
X_test_tr = X_test_normalised.transpose()
y_test_tr = y_test.reshape(1,y_test.shape[0])

print(X_train_tr.shape)
print(y_train_tr.shape)
print(X_test_tr.shape)
print(y_test_tr.shape)

dim_train = X_train_tr.shape[1]
dim_test = X_test_tr.shape[1]

print("The training dataset has dimensions equal to", dim_train)
print("The test set has dimensions equal to", dim_test)

y_train_shifted = y_train_tr - 1
y_test_shifted = y_test_tr - 1


print(y_train_shifted[:,1005])
print(y_train_shifted[:,1432])
print(y_train_shifted[:,456])
print(y_train_shifted[:,567])

Xtrain = X_train_tr
ytrain = y_train_shifted
Xtest = X_test_tr
ytest = y_test_shifted


logistic = LogisticRegression()

XX = x_train.T
YY = y_train.T.ravel()

print(XX)
print('---------------------------------------------------------------------')
print(YY)

print(logistic.fit(x_train,y_train))
print(logistic.score(XX,YY))
print(sum(logistic.predict(XX) == YY) / len(XX))