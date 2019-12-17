import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from matplotlib.pyplot import *

from scipy import interpolate
import pylab as pl

net1 = models.vgg16_bn(pretrained=True)
# print(1)
# net2 = models.alexnet(pretrained=True)1
# net3 = models.vgg16(pretrained=True)

def one():
    list_a = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
    for i in list_a:
        cov2d = net1.features[i].weight.data
        # print(len(cov2d))
        max_t = 0
        num = np.zeros(2500, dtype=int)
        for covfilter in cov2d:

            # print("filter:", covfilter)
            # for kernel in covfilter:
                # kernel_1d = kernel.flatten
                # norm = np.linalg.norm(kernel, ord='fro')
                # t = int(norm*1000)
                # print(t)
            weights = covfilter.flatten().flatten()
            for x in weights:
                t = int(x*1000)
                if t > max_t:
                    max_t = t
                num[t] = num[t]+1
        x = np.arange(0, max_t/1000, 0.001)
        f = interpolate.interp1d(x, num[0:max_t], kind='cubic')
        y = f(x)
        plot(x, y)
        title(i)

        savefig('{}.png'.format(i))
        show()

def three():
    list_c = [1, 4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]
    for i in list_c:
        bn2d = net1.features[i].weight.data
        # print(bn2d)
        max_t = 0
        num = np.zeros(1100, dtype=int)
        for gama in bn2d:
            t = int(gama*1000)
            if t > max_t:
                max_t = t
            num[t] = num[t] + 1
        # print(max_t)
        x = np.arange(0, max_t/1000, 0.001)
        f = interpolate.interp1d(x, num[0:max_t], kind='cubic')
        y = f(x)
        plot(x, y)
        title(i)
        savefig('{}.png'.format(i))
        show()
    # break

# def two():
#     list_b = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
#     list_norm_corr = []
#     list_norm_eudis = []
#     for i in list_b:
#         cov2d = net1.features[i].weight.data
#         filter_list = []
#         for covfilter in cov2d:
#             filter_t = covfilter.flatten().flatten().numpy().tolist()
#             filter_list.append(filter_t)
#         filter_np = np.array(filter_list)
#         # print(filter_np)
#         corr = np.corrcoef(filter_np)
#         # print(corr)
#         norm_corr = np.linalg.norm(corr, ord='fro')
#         print(i)
#         print(norm_corr)
#         list_norm_corr.append([i, norm_corr])
#         eudis_list = []
#         for line1 in filter_np:
#             tmp = []
#             for line2 in filter_np:
#                 x = np.linalg.norm(line1 - line2)
#                 tmp.append(x)
#             eudis_list.append(tmp)
#         norm_eudis = np.linalg.norm(np.array(eudis_list), ord='fro')
#         print(norm_eudis)
#         list_norm_eudis.append([i, norm_eudis])
#         # break
#
#     f1 = open('Euclidean_Distance.txt', 'a')
#     # f1.write(list_norm_eudis)
#     for x in list_norm_eudis:
#         f1.write(str(x[0]))
#         f1.write(' ')
#         f1.write(str(x[1]))
#         f1.write('\n')
#     f2 = open('Correlation_coefficient.txt', 'a')
#     for x in list_norm_corr:
#         f2.write(str(x[0]))
#         f2.write(' ')
#         f2.write(str(x[1]))
#         f2.write('\n')

def two():
    list_b = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
    list_norm_corr = []
    list_norm_eudis = []
    # print(net1.features[0].weight.data.numpy())
    # return
    layer = 1
    for i in list_b:
        cov2d = net1.features[i].weight.data
        filter_list = []
        for covfilter in cov2d:
            filter_t = covfilter.flatten().flatten().numpy().tolist()
            filter_list.append(filter_t)
        filter_np = np.array(filter_list)
        # print(filter_np)
        corr = np.corrcoef(filter_np)
        label = np.arange(0, len(corr))
        fig = figure()
        ax = fig.add_subplot(111)

        im = ax.imshow(corr, cmap=cm.hot_r)
        colorbar(im)
        title(layer)

        savefig('{}.png'.format(layer))
        layer += 1
        show()

        # break
        # max_t = 0
        # num = np.zeros(1001, dtype=int)
        # # corr = corr.flatten()
        # for a in range(len(corr)):
        #     for b in range(len(corr[a])):
        #         # print(corr[a][b])
        #         if a != b:
        #             t = abs(int(corr[a][b] * 1000))
        #             if t > max_t:
        #                 max_t = t
        #             num[t] += 1
        # x = np.arange(0, max_t / 1000, 0.001)
        # f = interpolate.interp1d(x, num[0:max_t], kind='cubic')
        # y = f(x)
        # # y = num[0:max_t]
        # plot(x, y)
        # title(i)
        # savefig('{}.png'.format(i))
        # show()

        # break
        # # print(corr)
        # norm_corr = np.linalg.norm(corr, ord='fro')
        # print(i)
        # print(norm_corr)
        # list_norm_corr.append([i, norm_corr])
        # eudis_list = []
        # for line1 in filter_np:
        #     tmp = []
        #     for line2 in filter_np:
        #         x = np.linalg.norm(line1 - line2)
        #         tmp.append(x)
        #     eudis_list.append(tmp)
        # norm_eudis = np.linalg.norm(np.array(eudis_list), ord='fro')
        # print(norm_eudis)
        # list_norm_eudis.append([i, norm_eudis])


def weight_cdf():
    list_a = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
    layer = 1
    for i in list_a:
        cov2d = net1.features[i].weight.data
        weights = cov2d.flatten().flatten().flatten().tolist()
        # temp = []

        temp2 = []
        for item in weights:
            if item != '':
                temp2.append(abs(float(item)))
        temp2.sort()
        # temp.append(temp2)
        # dataSets = temp

        plotDataset = [[], []]

        count = len(temp2)
        for i in range(count):
            plotDataset[0].append(float(temp2[i]))
            plotDataset[1].append((i + 1) / count)
        # print(plotDataset)
        plot(plotDataset[0], plotDataset[1], '-', linewidth=2)
        title(layer)
        savefig('{}.png'.format(layer))
        show()
        print(layer)
        # break
        layer += 1

def weight_cdf_all():
        weights = []
        list_a = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        for i in list_a:
            cov2d = net1.features[i].weight.data

            weights.extend(cov2d.flatten().flatten().flatten().tolist())
        # temp = []

        temp2 = []
        for item in weights:
            if item != '':
                temp2.append(abs(float(item)))
        temp2.sort()
        # temp.append(temp2)
        # dataSets = temp

        plotDataset = [[], []]

        count = len(temp2)
        for i in range(count):
            plotDataset[0].append(float(temp2[i]))
            plotDataset[1].append((i + 1) / count)
        # print(plotDataset)
        plot(plotDataset[0], plotDataset[1], '-', linewidth=2)
        title("all")
        savefig('{}.png'.format("all"))
        show()

def heatmap_weight():
    list_a = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
    layer = 1
    for i in list_a:
        cov2d = net1.features[i].weight.data
        # weights = cov2d.flatten().flatten().flatten().tolist()
        # temp = []
        weight = []
        for covfilter in cov2d:
            temp = covfilter.flatten().flatten().numpy().tolist()
            temp = [abs(i) for i in temp]
            weight.append(temp)

        weight = np.array(weight)
        label = np.arange(0, len(weight))
        fig = figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(weight, cmap=cm.hot_r)
        colorbar(im)
        title(layer)
        savefig('{}.png'.format(layer))
        show()
        print(layer)
        layer += 1

def heatmap_weight_all():
        weight = []
        list_a = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        for i in list_a:
            cov2d = net1.features[i].weight.data
            for covfilter in cov2d:
                temp = covfilter.flatten().flatten().numpy().tolist()
                temp = [abs(i) for i in temp]
                weight.append(temp)
                # print("append")

        # weight = np.array(weight)
        # print(weight)
        w2 = []
        for l in weight:
            w2.extend(l)
        w2 = np.array(w2)
        w2 = np.reshape(w2, (1728, 8513))
        print(w2)
        # return
        # print(w2.__len__())

        label = np.arange(0, len(w2))
        fig = figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(w2, cmap=cm.hot_r)
        colorbar(im)
        title("all")
        savefig('{}.png'.format("all"))
        show()





heatmap_weight_all()