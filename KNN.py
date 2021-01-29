import operator
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    """download iris data"""
    data, label = [], []
    iris = pd.read_csv('iris.data', header=None).values
    data = iris[:, 0:4]
    for i in range(len(iris)):
        if iris[i][4] == 'Iris-setosa':
            label.append([0])
        elif iris[i][4] == 'Iris-versicolor':
            label.append([1])
        elif iris[i][4] == 'Iris-virginica':
            label.append([2])
    label = np.array(label)
    label = label.reshape(150, 1)  # Transpose into a column of vectors
    return data, label


def shuffle(Z):
    """shuffle data"""
    index = [i for i in range(len(Z))]
    np.random.seed(20)
    np.random.shuffle(index)  # use Index to shuffle
    Z = Z[index, :]
    return Z


def main():
    X, Y = load_data()  # X,Y are data and label
    X = shuffle(X)
    Y = shuffle(Y)
    overall_accuracy = []
    K_list = []
    for K in range(1, 30):
        accuracy_sum = 0
        accuracy_avg = 0
        for j in range(5):
            x_train, y_train, x_test, y_test = split_data(j, X, Y)
            prediction = []
            for i in range(len(x_test)):  # 30
                x_1 = x_test[i]
                likely_class = KNN(x_1, x_train, y_train, K)
                prediction.append(likely_class)
            count = 0
            prediction = np.array(prediction)
            prediction = prediction.reshape(30, 1)
            for i in range(0, 30):
                if prediction[i, 0] == y_test[i, 0]:
                    count += 1
            accuracy = 100 * (count / 30)
            accuracy_sum = accuracy + accuracy_sum
            accuracy_avg = accuracy_sum / 5
        overall_accuracy.append(accuracy_avg)
        print("K=%d,The overall accuracy rate is：%.2f %%" % (K, accuracy_sum / 5))
        K_list.append(K)

    plt.plot(K_list, overall_accuracy)
    plt.xlabel("K")
    plt.ylabel("accuracy")
    plt.show()


def KNN(x_1, x_train, y_train, K):
    x_120 = np.tile(x_1, (len(x_train), 1))  # Copy x_test[0] to calculate delta (120,4)
    delta2 = (x_train - x_120) ** 2  # Calculate one dimensional distance (120,4)
    distance2 = []
    for i in range(len(delta2)):
        distance2.append(sum(delta2[i]))  # much like x^2+y^2, list(120,)
    distance = np.sqrt(distance2)

    sorted_distance = distance.argsort()  # sorts distance and returns the row index (120,)
    ClassCount = {}
    for i in range(K):  # Take the first K data
        class_K = y_train[sorted_distance[i]]
        class_K = class_K.tolist()[0]  # turn ndarray into int
        ClassCount[class_K] = (ClassCount.get(class_K, 0) + 1) / K
        # The occurrence times of various classes in k neighbors were counted.If there is no specific class_K, set new and value=0;If it has, value +1
    sorted_ClassCount = sorted(ClassCount.items(), key=operator.itemgetter(1), reverse=True)  # Sort items from large to small by the second field of dict
    likely_class = sorted_ClassCount[0][0]  # return 0，1 or 2
    return likely_class


def split_data(j, X, Y):
    if j == 0:
        x_train = X[0:120, :]
        y_train = Y[0:120, :]
        x_test = X[120:150, :]
        y_test = Y[120:150, :]
    elif j == 1:
        x_train = X[30:150, :]
        y_train = Y[30:150, :]
        x_test = X[0:30, :]
        y_test = Y[0:30, :]
    elif j == 2:
        x_train = np.vstack((X[0:30, :], X[60:150, :]))
        y_train = np.vstack((Y[0:30, :], Y[60:150, :]))
        x_test = X[30:60, :]
        y_test = Y[30:60, :]
    elif j == 3:
        x_train = np.vstack((X[0:60, :], X[90:150, :]))
        y_train = np.vstack((Y[0:60, :], Y[90:150, :]))
        x_test = X[60:90, :]
        y_test = Y[60:90, :]
    elif j == 4:
        x_train = np.vstack((X[0:90, :], X[120:150, :]))
        y_train = np.vstack((Y[0:90, :], Y[120:150, :]))
        x_test = X[90:120, :]
        y_test = Y[90:120, :]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    main()