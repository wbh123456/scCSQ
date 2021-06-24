# direct import  hadamrd matrix from scipy
from scipy.linalg import hadamard  
import torch
from torchvision import models
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import os
from tqdm import tqdm
from collections import Counter
import statistics
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import shutil
import fairscale
import time

# use algorithm 1 to generate hash centers
def get_hash_centers(n_class, bit):
    H_K = hadamard(bit)
    H_2K = np.concatenate((H_K, -H_K), 0)
    hash_targets = torch.from_numpy(H_2K[:n_class]).float()

    if H_2K.shape[0] < n_class:
        hash_targets.resize_(n_class, bit)
        for k in range(20):
            for index in range(H_2K.shape[0], n_class):
                ones = torch.ones(bit)
                # Bernouli distribution
                sa = random.sample(list(range(bit)), bit // 2)
                ones[sa] = -1
                hash_targets[index] = ones
            # to find average/min  pairwise distance
            c = []
            for i in range(n_class):
                for j in range(n_class):
                    if i < j:
                        TF = sum(hash_targets[i] != hash_targets[j])
                        c.append(TF)
            c = np.array(c)

            # choose min(c) in the range of K/4 to K/3
            # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
            # but it is hard when bit is  small
            if c.min() > bit / 4 and c.mean() >= bit / 2:
                print(c.min(), c.mean())
                break
    return hash_targets

# 计算所有metrics的top-level interface
def compute_metrics(query_dataloader, net, class_num, show_time=False, use_cpu=False):
      ''' Labeling Strategy:
      Closest Hash Center:
      Label the query using the label associated to the nearest hash center
      - Computation Complexity:
        O(m) << O(n) per puery
        m = number of classes in database
      - Less Accurate
      '''
      start_time_CHC = time.time()
      if use_cpu:
        binaries_query, labels_query = compute_result_cpu(query_dataloader, net)
      else:
        binaries_query, labels_query = compute_result(query_dataloader, net)

      # 根据自定义的labeling策略，得到预测的labels
      hash_centers = net.forward_hash_center().sign()
      labels_pred_CHC = get_labels_pred_closest_hash_center(binaries_query.cpu().numpy(), labels_query.numpy(),
                                                            hash_centers.cpu().numpy())
      CHC_duration = time.time() - start_time_CHC
      query_num = binaries_query.shape[0]
      if show_time:
        print("  - CHC query speed with {} test data: {:.2f} queries/s".format(query_num, query_num/CHC_duration))
      
      # (1) 自定义的labeling策略的accuracy
      labeling_accuracy_CHC = compute_labeling_strategy_accuracy(labels_pred_CHC, labels_query.numpy())
      
      # (2) F1_score, average = (micro, macro, weighted)
      F1_score_weighted_average_CHC = f1_score(labels_query, labels_pred_CHC, average='weighted')
      F1_score_per_class_CHC = f1_score(labels_query, labels_pred_CHC, average=None)

      # (3) F1_score median
      F1_score_median_CHC = statistics.median(F1_score_per_class_CHC)

      CHC_metrics = (labeling_accuracy_CHC, F1_score_weighted_average_CHC, F1_score_median_CHC, F1_score_per_class_CHC)

      return CHC_metrics

def euclidean_distance(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x (tensor): N x D
    # y (tensor): M x D
    # Return: N x M
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception("size mismatch")

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    # l = torch.nn.BCELoss(reduction="none")
    # loss = l(x.float(), y.float())

    # return loss.mean(2)
    return torch.pow(x - y, 2).sum(2)

def make_one_hot(labels, n_classes):
    """
    :param labels: the labels in int form
    :param n_classes: the number of classes
    :return: a one hot vector with these class labels
    """
    one_hot = torch.zeros((labels.shape[-1], n_classes))
    return one_hot.scatter_(1, torch.unsqueeze(labels, 1).long().cpu(), 1).float()

def test_speed(dataloader, net, size=280):
    # get 200 data samples and evaluate them
    data_200, labels_200 = [], []
    net.cpu()
    net.eval()
    for img, label in dataloader:
        if len(data_200) > 0 and data_200[0].shape[0] * len(data_200) > size:
            break
        data_200.append(img)
        labels_200.append(label)
    data_200 = torch.cat(data_200)[:size]
    labels = torch.cat(labels_200)[:size]

    # Take the average of 6 runs as to calculate query speed
    times = []
    rep_num = 6
    for i in range(rep_num):
        start_time_CHC = time.time()
        binary_codes = (net(data_200.cpu())).data
        hash_centers = net.forward_hash_center().cpu().data
        labels_pred_CHC = get_labels_pred_closest_hash_center(binary_codes.cpu().numpy(), labels.numpy(), hash_centers.numpy())
        CHC_duration = time.time() - start_time_CHC
        times.append(CHC_duration)
    times = np.array(times)
    query_num = binary_codes.shape[0]
    print("  - Average CHC Query Speed with {} test data: {:.2f} queries/s, time = {:.2f}".format(query_num, query_num/times.mean(), times.mean()))
    print("  - Each time =", times)


# 了解Top K原理的链接：https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
def compute_MAP(retrieval_binaries, query_binaries, retrieval_labels, query_labels, topk):
    num_query = query_labels.shape[0]
    topK_ave_precision_per_query = 0
    for iter in range(num_query):
        # 对于一个query label，查看database实际有多少相同的labels Ex: [1,0,0,0,1,1,1,1,0,0]
        ground_truths = (np.dot(query_labels[iter,:], retrieval_labels.transpose()) > 0).astype(np.float32)

        # Given a query binary，计算其与其他database里所有的binaries的hamming distance Ex: [2,10,14,9,1,2,1,2,1,4,6]
        hamm_dists = CalcHammingDist(query_binaries[iter, :], retrieval_binaries)
        
        # 根据从小到的hamming distance，返回对应的index
        hamm_indexes = np.argsort(hamm_dists)

        # 理想情况下: [1,1,1,1,1,0,0,0,0,0]
        # index对应的hamming distance: [1,1,1,2,2,4,6,9,10,14]
        ground_truths = ground_truths[hamm_indexes]

        # topk的选择可能也会有不小的影响。。。
        topK_ground_truths = ground_truths[0:topk]

        # Ex: topK_ground_truths = 5
        topK_ground_truths_sum = np.sum(topK_ground_truths).astype(int)

        # 问题：如果database里面没有query的label咋办。。。？
        if topK_ground_truths_sum == 0:
          #******需不需要 num_query -= 1
            continue

        # Ex: [1,2,3,4,5]
        matching_binaries = np.linspace(1, topK_ground_truths_sum, topK_ground_truths_sum)

        # ground truths position范围在1 ~ n
        ground_truths_pos = np.asarray(np.where(topK_ground_truths == 1)) + 1.0

        topK_ave_precision_per_query_ = np.mean(matching_binaries / (ground_truths_pos))

        topK_ave_precision_per_query += topK_ave_precision_per_query_
        
    topK_map = topK_ave_precision_per_query / num_query

    return topK_map


# Predict label using Closest Hash Center strategy (b)
def get_labels_pred_closest_hash_center(query_binaries, query_labels, hash_centers):
    num_query = query_labels.shape[0]
    labels_pred = []
    for binary_query, label_query in zip(query_binaries, query_labels):
          dists = CalcHammingDist(binary_query, hash_centers)
          closest_class = np.argmin(dists)
          labels_pred.append(closest_class)
    return labels_pred


# 简单比较query和pred labels的相同个数并算一个accuracy
def compute_labeling_strategy_accuracy(labels_pred, labels_query):
    same = 0

    for i in range(len(labels_pred)):
      if (labels_pred[i] == labels_query[i]).all():
        same += 1

    return same / labels_query.shape[0]


# 计算Binary和得到labels
def compute_result(dataloader, net):
    binariy_codes, labels = [], []
    net.eval()
    for img, label in dataloader:
        labels.append(label)
        binariy_codes.append((net(img.cuda())).data)
    return torch.cat(binariy_codes).sign(), torch.cat(labels)

# 计算Binary和得到labels
def compute_result_cpu(dataloader, net):
    binariy_codes, labels = [], []
    net.cpu()
    net.eval()
    for img, label in dataloader:
        labels.append(label.cpu())
        binariy_codes.append((net(img.cpu())).data)
    return torch.cat(binariy_codes).sign(), torch.cat(labels)


# 计算hamming distance，B1是一组data，（一个vector），B2是一个matrix（所有database里的vector）
def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


# 找到重复次数最多的label，还没有考虑怎么break even，或者根据rank来assign不同的weight
def find_most_common_label(labels):
  # 为了hash，变成tuple
  labels_tuple = [tuple(label) for label in labels]

  most_common_label_tuple = Counter(labels_tuple).most_common(1)[0][0]
  return np.array(most_common_label_tuple)


# 把提供的categorical labels转换成one-hot形式
def categorical_to_onehot(labels, numOfClass):
    labels = labels.reshape(labels.shape[0], 1)
    labels = (labels == torch.arange(numOfClass).reshape(1, numOfClass)).int()
    return labels

# T-sne visualizations
def t_sne_visualization_2d(data, labels):
  from sklearn.manifold import TSNE
  from matplotlib import pyplot as plt
  import matplotlib.cm as cm

  # Calculate TSNE embeded space
  embedded_data = TSNE(n_components=2,
                       n_iter=5000, 
                       learning_rate=100,
                       early_exaggeration=10,
                       perplexity=50
                       ).fit_transform(data)

  # Visualization 
  plt.figure(figsize=(6, 5))

  # unique_classes = set(labels)
  # colors = cm.rainbow(np.linspace(0, 1, len(unique_classes)))
  
  # for label, c in zip(unique_classes, colors):
  #     plt.scatter(embedded_data[labels == label, 0], 
  #                 embedded_data[labels == label, 1], 
  #                 linewidths=0.03,
  #                 color=c, 
  #                 label=label)
      
  plt.scatter(x = embedded_data[:,0], y = embedded_data[:,1], c = labels, s = 1.5, cmap='viridis')
  plt.title('t-SNE visualization of hash code generation - TM datasets')

  plt.legend()
  plt.show()
  return
