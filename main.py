# coding=utf-8
# 在0-2*pi的区间上生成100个点作为输入数据
import numpy as np
from matplotlib import pyplot as pt
from PIL import Image
import matplotlib.pyplot as plt
import math
import random

def create_data():
    n = 1000
    k = 3
    mu = [0,-0.5,0.3]
    sigma = [0.1,0.05,0.07]
    number = [400,300,300]
    data = []
    for i in range(len(number)):
        data.append(np.random.normal(mu[i], sigma[i], number[i]).tolist())
        # count, bins3, ignored = plt.hist(data[i], 30, density=True)
    # count1, bins1, ignored1 = plt.hist(data, 30, density=True)
    # plt.plot(bins1, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins1 - mu1) ** 2 / (2 * sigma1 ** 2)), linewidth=2,
    #        color='r')
    #plt.hist(data, 50)
    #plt.title("GMM")
    #plt.show()
    return data

def count_q(K,N,gama,sigma,y,mu,alpha):
    '''
    计算Q函数该轮迭代以后的取值，其中Q函数太长了，分成了Q前部和Q后部两部分
    :param K:模型个数
    :param N:样本个数
    :param gama:
    :param sigma:
    :param y:数据
    :param mu:
    :param alpha:
    :return:
    '''
    gsum = []
    for k in range(K):
        gsum.append(np.sum([gama[j, k] for j in range(N)]))
    for g,ak in zip(gsum,alpha):
        q_first = np.sum(g * np.log(ak))  # Question
    q_sencond = np.sum([np.sum([gama[j, k] * (
                np.log(1 / np.sqrt(2 * np.pi)) - np.log(np.sqrt(sigma[k])) - (1 / (2 * sigma[k]) * (y[j] - mu[k]) ** 2))
                    for j in range(N)]) for k in range(K)])
    return q_first+q_sencond

def EM(iter,Epsilon,N,K,data):
    '''

    :param iter: 迭代次数
    :param Epsilon: 阈值
    :param N:
    :param K:
    :param data: 数据集
    :return: 返回算出的参数
    '''
    alpha = np.ones(K) # 初始alpha
    mu = [0,1,2]
    print("mu",mu)
    sigma = [10] * K
    print("sigama",sigma)
    gama = np.ones((N, K))
    q1 = 0
    for i in range(iter):
        for k in range(K):
            # 计算模型k :gama
            for j in range(N):
                fx = alpha[k]*(np.exp(-(data[j]-mu[k])**2/(2*sigma[k])))/(np.sqrt(2*np.pi*sigma[k]))
                gama[j,k] = fx/np.sum([(alpha[k]*(np.exp(-(data[j]-mu[k])**2/(2*sigma[k])))/(np.sqrt(2*np.pi*sigma[k]))) for k in range(K)])

        for k in range(K):
            # 计算模型k的 :mu sigma alpha
            mu[k] = (np.sum([ gama[j,k]*data[j] for j in range(N)]))/(np.sum([gama[j,k] for j in range(N)]))
            sigma[k] =(np.sum([ gama[j,k]*(data[j]-mu[k])**2 for j in range(N)]))/(np.sum([gama[j,k] for j in range(N)]))
            alpha[k] = (np.sum([gama[j,k] for j in range(N)]))/N
        q2 = count_q(K, N, gama, sigma, data, mu, alpha)
        # print(abs(q2 - q1))
        # 小于某个阈值则退出
        if abs(q2 - q1)< Epsilon:
            break
        q1 = q2
    return alpha,mu,sigma

def draw_picture(mu, sigma,data):
    bins = 100
    x = [0,0,0]
    color = ["green","black","blue"]
    for k in range(len(mu)):
        y = (np.exp(-(data-mu[k])**2/(2*sigma[k])))/(np.sqrt(2*np.pi*sigma[k]))
        plt.scatter(data, y, c='red', marker='x')
    plt.title("EM_GMM")
    plt.show()

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
    return np.mat(list_tolist)

if __name__ == '__main__':
    N = 1000 # 样本的个数
    K = 3 # 有个高斯模型
    Epsilon = 0.001
    Iter = 1000
    # data_array = create_data()
    #data = []
    #for i in range(K):
     #   data = data_array[i]+data
    data = loan_data("data.txt")
    alpha_result, mu_result, sigma_result = EM(Iter, Epsilon, N, K, data)
    print("result:",alpha_result,mu_result,sigma_result)

    draw_picture(mu_result, sigma_result,data)
