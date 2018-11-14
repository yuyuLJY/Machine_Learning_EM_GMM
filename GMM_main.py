# coding=utf-8
import numpy as np
from matplotlib import pyplot as pt
from PIL import Image
import matplotlib.pyplot as plt
import math
import random

def loan_data(txt_name):
    '''
    读入数据
    :param txt_name:
    :return:数据矩阵
    '''
    data = open(txt_name).readlines()
    length = len(data)
    list_tolist = []
    for line in data:
        line = line.strip().split(' ')
        for i in range(2):
            line[i] = float(line[i])
        list_tolist.append(line)
    return list_tolist

def density_probability(x,mean ,sigma):
    print(x,mean)
    print("x-mean",np.shape(x-mean))
    print("x-mean-T:",np.shape(np.transpose(x-mean)))
    print("sigmaI:",sigma.I)
    a = np.dot(np.transpose(x-mean),sigma.I)
    up = np.exp(-1/2*np.dot(a,x-mean))
    down = np.power(2*np.pi, col/2) * np.power(np.linalg.det(sigma), 1/2)
    print("pro:", up/down)
    return up/down

def init(data,K):
    # 仅仅针对二维
    row, col = np.shape(data)
    alpha = []
    mean = []
    sigma = [np.mat([[0.1, 0], [0, 0.1]]) for x in range(K)]
    for i in range(K):
        alpha.append(1/3)
        seed = random.randint(0, col-1)
        print(seed)
        mean.append(data[:,seed])  # 截取某一列
    return alpha,mean,sigma


def EM(data,K,iter):
    row, col = np.shape(data)
    # 初始化means sigma alpha
    alpha, mean, sigma = init(data,K)
    print(alpha, mean, sigma)
    gamma = np.mat(np.zeros((col, 3)))
    # 计算gamma
    for i in range(iter):
        for j in range(col):
            sum = 0
            for k in range(K):
                gamma[j, k] = alpha[k] * density_probability(data[:, j], mean[k], sigma[k])
                sum += gamma[j, k]
            for k in range(K):
                gamma[j, k] = gamma[j, k] / sum  # 已经计算完每一个gamma
                # print(gamma[j, k])
        # rr,cc = np.shape(gamma)
        # print(rr,cc) # 270 3
        gamma_sum = np.sum(gamma, axis=0) # 1*3
        # print("gamma_sum:",gamma)

         # 计算mean sigma alpha
        for k in range(K):
            for j in range(col):
                mean[k] += gamma[j, k] * data[:, j]
            mean[k] = mean[k] / gamma_sum[:,k]

            for j in range(col):
                sigma[k] += gamma[j, k] * np.dot((data[:,j] - mean[k]) ,np.transpose(data[:,j] - mean[k]))
            sigma[k] = sigma[k] / gamma_sum[:,k]

            alpha[k] = gamma_sum [:,k]/ col

    #对gama进行提取特征
    index = []
    for m in range(col):
        index.append(gamma[m,:].argmax(axis=0))
    return index



if __name__ == '__main__':
    K = 3
    iter = 100
    data = loan_data("data.txt") # 120*2的矩阵
    data_Transpose = np.transpose(data)
    # print(data_Transpose)  # 变成 2*120
    row, col = np.shape(data_Transpose) # row =2,col = 270
    print(row,col)
    index = EM(data_Transpose, K, iter)
    print(index)