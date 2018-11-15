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
    mean = mean.T
    up = np.exp(-1/2*np.dot(np.dot(np.transpose(x-mean),sigma.I),x-mean))
    down = 2*np.pi * np.power(np.linalg.det(sigma), 1/2)
    return np.sum(up)/np.sum(down)

def init(data,K):
    # 仅仅针对二维
    row, col = np.shape(data)
    mean = np.mat([[0.3617 ,1.8347],[-0.4479, -1.3303],[1.0784 ,5.3157]])
    sigma = [np.mat([[1, 0], [0, 1]]) for x in range(K)]
    alpha = [1/3,1/3,1/3]
    return alpha,mean,sigma


def EM(data,K,iter):
    row, col = np.shape(data)
    alpha, mean, sigma = init(data,K)    # 初始化means sigma alpha
    gamma = np.mat(np.zeros((col, 3)))
    for i in range(iter):
        for j in range(col):
            sum = 0
            for k in range(K):
                gamma[j, k] = alpha[k] * density_probability(data[:, j], mean[k], sigma[k])
                sum += gamma[j, k]
            for k in range(K):
                    gamma[j, k] = gamma[j, k] / float(sum)  # 已经计算完每一个gamma
        #print("gamma:", gamma)
        gamma_sum = np.sum(gamma, axis=0) # 1*3
        for k in range(K):
            # mean[k] = np.zeros((1,row)) # 开始的时候忘记写了，找bug很痛苦
            mean[k] = np.mat([0.0,0.0])
            sigma[k] = np.mat([[0.0, 0.0], [0.0, 0.0]])
            for j in range(col):
                mean[k] += gamma[j, k] * data[:, j].T
                # print("sigma[k]", mean[k])
            mean[k] = mean[k] / gamma_sum[0,k]

            for j in range(col):
                sigma[k] += gamma[j, k] * np.dot((data[:,j] - mean[k].T) ,np.transpose(data[:,j] - mean[k].T))
            sigma[k] = sigma[k] / gamma_sum[0,k]

            alpha[k] = gamma_sum [0,k]/ col
        print("sigma:",sigma)
        print("mean",mean)
        print("alpha",alpha)
    return gamma

def classify(gamma,data):
    '''
    根据算出的gamma分类
    :param gamma:
    :param data:
    :return:聚类中心和聚类情况的标签
    '''
    row, col = np.shape(gamma)
    clusterAssign = np.mat(np.zeros((row, 2)))
    for i in range(row):
        # amx返回矩阵最大值，argmax返回矩阵最大值所在下标
        clusterAssign[i, :] = np.argmax(gamma[i, :]), np.amax(gamma[i, :])  # 15.确定x的簇标记lambda
    data_toarray = np.mat(data).A
    clustercents = np.mat(np.ones((col, 2)))  # 创建 (3*2)行的中心点
    for cent in range(col):
        cent_index = np.nonzero(clusterAssign[:, 0] == cent)[0]  # 得到同一个聚类的所有索引
        cent_data = data_toarray[cent_index]
        if len(cent_data!=0): # 不是空的才求平均数
            clustercents[cent, :] = np.mean(cent_data, axis=0)  # 对列求均值
    return clustercents,clusterAssign

def draw_picture(clustercents,ClustDist,data,k):
    ClustDist_toarray = ClustDist.A
    dataSet_toarray = np.mat(data).A
    color = ['blue', 'yellow', 'green', 'black', 'c']  # g’‘b’‘c’‘m’‘y’‘k’
    for cent in range(k):
        index = np.nonzero(ClustDist_toarray[:, 0] == cent)[0]
        cluster = dataSet_toarray[index]
        plt.scatter(cluster[:, 0], cluster[:, 1], c=color[cent], marker='o')
        plt.scatter(clustercents[cent, 0], clustercents[cent, 1], c='red', marker='x')
    plt.show()

if __name__ == '__main__':
    K = 3
    iter = 8
    data = loan_data("data.txt") # 120*2的矩阵
    data_Transpose = np.transpose(data)
    # print(data_Transpose)  # 变成 2*120
    row, col = np.shape(data_Transpose) # row =2,col = 270
    gamma = EM(np.mat(data_Transpose), K, iter)
    clustercents, ClustDist = classify(gamma, data) # 进行分割
    draw_picture(clustercents,ClustDist,data,K)