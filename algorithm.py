# MLP之演算法，包含train以及test

# 原生python library
import math
# 另外下載之python library
import numpy as np


# here use sigmoid function
def activation_func(v):
    return 1 / (1 + math.exp(-v))


# derivative of sigmoid function
def derivative_activation_func(v):
    return activation_func(v) * (1 - activation_func(v))


# a為學習率, it為iteration, es為early stop(以MSE做為依據)
# 1層隱藏層, 隱藏層裡有2個cell, 輸出層也有2個cell，只能分兩類
def mlp_train(x, y, anskey, lr=0.1, it=100, es=0.1, num_neuron=4):
    # 初始化部分數值與weight
    row = len(x)
    col = len(x[0])
    # 兩層隱藏層架構
    w1 = []
    for temp in range(int(num_neuron)):
        w1.append(np.random.uniform(low=-1, high=1, size=col))
    w2 = []
    for temp in range(int(num_neuron)):
        w2.append(np.random.uniform(low=-1, high=1, size=len(w1) + 1))
    # 輸出層
    w3 = []
    for temp in range(2):  # 這邊應設為len(anskey), but目前僅能處理2群
        w3.append(np.random.uniform(low=-1, high=1, size=len(w2) + 1))
    E, accuracy, expected_answer = mlp_test(x, y, [w1, w2, w3], anskey=anskey)  # 初始化E和accuracy

    # 訓練開始
    for i in range(it):
        print('Epoch ' + str(i + 1))
        # 調整階段
        for n in range(row):  # 有幾列就代表有多少筆資料
            # forwarding
            x1 = x[n]
            x2 = []
            for temp in range(len(w1)):
                x2.append(neuron_output(w1[temp], x1))
            x2 = np.asarray(x2)
            x2 = np.append(-1, x2)
            x3 = []
            for temp in range(len(w2)):
                x3.append(neuron_output(w2[temp], x2))
            x3 = np.asarray(x3)
            x3 = np.append(-1, x3)
            output = []
            for temp in range(len(w3)):
                output.append(neuron_output(w3[temp], x3))
            # back propagation
            # 計算delta(δ)
            delta3 = []
            if y[n] == anskey[0]:
                delta3.append(output_delta(1, output[0]))
                delta3.append(output_delta(0, output[1]))
            elif y[n] == anskey[1]:
                delta3.append(output_delta(0, output[0]))
                delta3.append(output_delta(1, output[1]))
            else:
                delta3.append(0)
                delta3.append(0)
            delta2 = []
            for temp in range(len(w2)):
                delta2.append(internal_delta(x3[temp + 1], np.append(delta3[0], delta3[1]),
                                             np.append(w3[0][temp + 1], w3[1][temp + 1])))
            delta1 = []
            for temp in range(len(w1)):
                delta1.append(internal_delta(x2[temp + 1], np.append(delta2[0], delta2[1]),
                                             np.append(w2[0][temp + 1], w2[1][temp + 1])))
            # 調整鍵結值
            for temp in range(len(w1)):
                w1[temp] = adjust_w(w1[temp], delta1[temp], x1, iota=lr)
            for temp in range(len(w2)):
                w2[temp] = adjust_w(w2[temp], delta2[temp], x2, iota=lr)
            for temp in range(len(w3)):
                w3[temp] = adjust_w(w3[temp], delta3[temp], x3, iota=lr)
        # 驗證階段
        E, accuracy, expected_answer = mlp_test(x, y, [w1, w2, w3], anskey=anskey)
        print('Mean square error is ' + str(E) + '\nAccuracy is ' + str(accuracy))
        if E < es:
            print('Early stop at epoch ' + str(i + 1))
            return [w1, w2, w3], E, accuracy, expected_answer
    return [w1, w2, w3], E, accuracy, expected_answer


def mlp_test(x, y, w, anskey):
    E = 0
    correct = 0
    expected_answer = []
    row = len(x)
    for n in range(row):  # 驗證所有資料，看有幾列，代表有多少筆資料
        x1 = x[n]
        x2 = []
        for temp in range(len(w[0])):
            x2.append(neuron_output(w[0][temp], x1))
        x2 = np.asarray(x2)
        x2 = np.append(-1, x2)
        x3 = []
        for temp in range(len(w[1])):
            x3.append(neuron_output(w[1][temp], x2))
        x3 = np.asarray(x3)
        x3 = np.append(-1, x3)
        output = []
        for temp in range(len(w[2])):
            output.append(neuron_output(w[2][temp], x3))
        # 判斷輸出為分類1或分類2
        if output[0] >= output[1]:
            expected_answer.append(anskey[0])
        else:
            expected_answer.append(anskey[1])
        if y[n] == anskey[0]:  # 答案為第一類
            if output[0] >= output[1]:  # 判斷為第一類
                correct += 1
            E = E + (1 / 2) * math.pow(output[0] - 1, 2)
        elif y[n] == anskey[1]:  # 答案為第二類
            if output[0] < output[1]:  # 判斷為第二類
                correct += 1
            E = E + (1 / 2) * math.pow(output[1] - 1, 2)
        else:  # 非第一類或第二類
            E = E + (1 / 2) * math.pow(output[0] + output[1], 2)
    accuracy = correct / row
    return E, accuracy, expected_answer


def neuron_output(w, x):
    return activation_func(np.dot(w, x))


def output_delta(d, o):
    return (d - o) * o * (1 - o)


def internal_delta(y, delta, w):
    return y * (1 - y) * np.dot(delta, w)


def adjust_w(w, delta, x, iota=0.8):
    return w + iota * delta * x
