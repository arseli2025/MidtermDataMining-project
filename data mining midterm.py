#Arsel Curri classification of three simple features in a dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
iris = datasets.load_iris()
X = iris.data[:, :2] 
y = iris.target
y_cat = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train.argmax(axis=1))  
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu')) 
model.add(Dense(3, activation='softmax'))  
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
def plot_decision_boundary(X, y, model=None, clf=None, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))    
    Z = None
    if model is not None:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(Z, axis=1)
    elif clf is not None:
        # Use the KNN model
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=plt.cm.coolwarm)
    plt.title(title)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_decision_boundary(X, y, clf=knn)
plt.title("KNN Classifier Decision Boundary")
plt.subplot(1, 2, 2)
plot_decision_boundary(X, y, model=model)
plt.title("Deep Learning Model Decision Boundary")
plt.show()
