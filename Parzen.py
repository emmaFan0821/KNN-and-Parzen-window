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
    XA, XB, XC, YA, YB, YC = data_class(X, Y)  # (150,4)
    h_list = []
    h = 0
    for n in range(20):
        h += 0.1
        h_list.append(h)
    overall_accuracy = []
    for h in h_list:
        accuracy_sum = 0
        for j in range(5):
            x_train, y_train, x_valid, y_valid = split_data(j, XA, XB, XC, YA, YB, YC)
            x_valid = shuffle(x_valid)
            y_valid = shuffle(y_valid)
            XY = np.hstack((x_train, y_train))
            XY = shuffle(XY)

            XY = XY[np.lexsort(XY.T)]  # Sort by the fifth column, y
            prediction = []
            for k in range(len(x_valid)):  # 24
                x_1 = x_valid[k]  # (1x4) vector to be classified
                P = []
                for l in range(3):
                    x_train = XY[0 + 32 * l:32 + 32 * l, 0:4]  # The probabilities are calculated separately for each category
                    Pwkx = parzen(x_1, x_train, h)
                    P.append(Pwkx)
                likely_class = P.index(max(P))
                prediction.append(likely_class)

            count = 0
            prediction = np.array(prediction)
            prediction = prediction.reshape(24, 1)
            for i in range(0, 24):
                if prediction[i, 0] == y_valid[i, 0]:
                    count += 1
            accuracy = 100 * (count / 24)
            accuracy_sum = accuracy + accuracy_sum
            # print("The accuracy of number %d testing set is：%.2f %%" % (j + 1, accuracy))
        accuracy_avg = accuracy_sum / 5
        overall_accuracy.append(accuracy_avg)
        print("h = %f, The overall validation accuracy rate is：%.2f %%" % (h, accuracy_avg))
    plt.plot(h_list, overall_accuracy)
    plt.xlabel("h")
    plt.ylabel("accuracy")
    plt.show()

    """进行测试"""
    x_test = np.vstack((XA[40:50, :], XB[40:50, :], XC[40:50, :]))
    y_test = np.vstack((YA[40:50, :], YB[40:50, :], YC[40:50, :]))
    x_test = shuffle(x_test)
    y_test = shuffle(y_test)
    prediction = []
    for k in range(len(x_test)):  # 30
        x_1 = x_test[k]
        P = []
        for l in range(3):
            x_train = XY[0 + 32 * l:32 + 32 * l, 0:4]
            Pwkx = parzen(x_1, x_train, h=0.1)
            P.append(Pwkx)
        likely_class = P.index(max(P))
        prediction.append(likely_class)
    count = 0
    prediction = np.array(prediction)
    prediction = prediction.reshape(30, 1)
    print("Prediction of test data: ", prediction.T)
    print("y_test: ", y_test.T)
    for i in range(0, 30):
        if prediction[i, 0] == y_test[i, 0]:
            count += 1
    accuracy = 100 * (count / 30)
    print("The test dataset accuracy rate is：%.2f %%" % accuracy)


def parzen(x_1, x_train, h):
    nk = x_train.shape[0]
    x_32 = np.tile(x_1, (len(x_train), 1))
    delta2 = (x_train - x_32) ** 2
    distance2 = []
    for m in range(len(delta2)):
        distance2.append(sum(delta2[m]))
    distance = np.sqrt(distance2)

    u = distance / h
    phi_u = np.exp(u**2 / -2) / np.sqrt(2 * np.pi)
    p = np.sum(phi_u / h**4) / nk
    Pwkx = p / 3  # Pwkx[i] = Pwk * Pxwk
    return Pwkx


def data_class(X, Y):
    XA = X[0:50, 0:4]
    XB = X[50:100, 0:4]
    XC = X[100:150, 0:4]
    YA = Y[0:50, 0:4]
    YB = Y[50:100, 0:4]
    YC = Y[100:150, 0:4]
    return XA, XB, XC, YA, YB, YC


def split_data(j, XA, XB, XC, YA, YB, YC):
    if j == 0:
        x_valid = np.vstack((XA[0:8, :], XB[0:8, :], XC[0:8, :]))
        y_valid = np.vstack((YA[0:8, :], YB[0:8, :], YC[0:8, :]))
        x_train = np.vstack((XA[8:40, :], XB[8:40, :], XC[8:40, :]))
        y_train = np.vstack((YA[8:40, :], YB[8:40, :], YC[8:40, :]))
    elif j == 1:
        x_valid = np.vstack((XA[8:16, :], XB[8:16, :], XC[8:16, :]))
        y_valid = np.vstack((YA[8:16, :], YB[8:16, :], YC[8:16, :]))
        x_train = np.vstack((XA[16:40, :], XA[0:8, :], XB[16:40, :], XB[0:8, :], XC[16:40, :], XC[0:8, :]))
        y_train = np.vstack((YA[16:40, :], YA[0:8, :], YB[16:40, :], YB[0:8, :], YC[16:40, :], YC[0:8, :]))
    elif j == 2:
        x_valid = np.vstack((XA[16:24, :], XB[16:24, :], XC[16:24, :]))
        y_valid = np.vstack((YA[16:24, :], YB[16:24, :], YC[16:24, :]))
        x_train = np.vstack((XA[24:40, :], XA[0:16, :], XB[24:40, :], XB[0:16, :], XC[24:40, :], XC[0:16, :]))
        y_train = np.vstack((YA[24:40, :], YA[0:16, :], YB[24:40, :], YB[0:16, :], YC[24:40, :], YC[0:16, :]))
    elif j == 3:
        x_valid = np.vstack((XA[24:32, :], XB[24:32, :], XC[24:32, :]))
        y_valid = np.vstack((YA[24:32, :], YB[24:32, :], YC[24:32, :]))
        x_train = np.vstack((XA[32:40, :], XA[0:24, :], XB[32:40, :], XB[0:24, :], XC[32:40, :], XC[0:24, :]))
        y_train = np.vstack((YA[32:40, :], YA[0:24, :], YB[32:40, :], YB[0:24, :], YC[32:40, :], YC[0:24, :]))
    elif j == 4:
        x_valid = np.vstack((XA[32:40, :], XB[32:40, :], XC[32:40, :]))
        y_valid = np.vstack((YA[32:40, :], YB[32:40, :], YC[32:40, :]))
        x_train = np.vstack((XA[0:32, :], XB[0:32, :], XC[0:32, :]))
        y_train = np.vstack((YA[0:32, :], YB[0:32, :], YC[0:32, :]))
    return x_train, y_train, x_valid, y_valid


if __name__ == '__main__':
    main()
