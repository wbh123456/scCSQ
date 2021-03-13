# direct import  hadamrd matrix from scipy
from scipy.linalg import hadamard  
import torch
from torchvision import models
from torch import nn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import os
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import Counter
import statistics
from torchvision.datasets.utils import download_and_extract_archive
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score
# from ray.tune.integration.pytorch_lightning import TuneReportCallback
# from ray import tune
# from ray.tune import CLIReporter
import shutil
import fairscale

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

# Needs to add algorithm 2 to generate hash centers



# 计算所有metrics的top-level interface
def compute_metrics(database_dataloader, query_dataloader, net, top_k, class_num):
      ''' Labeling Strategy:

      (a) Majority Vote:  
      Label the query using the majority labels in the top K nearest 
      (in hamming space) cells in the database.
      - Computation Complexity:
        O(nlogK) per query   --> Current implementation O(nlogn)
        n = number of cells in database
      - Very Accurate

      (b) Closest Hash Center:
      Label the query using the label associated to the nearest hash center
      - Computation Complexity:
        O(m) << O(n) per puery
        m = number of classes in database
      - Less Accurate
      '''

      binaries_database, labels_database = compute_result(database_dataloader, net)
      binaries_query, labels_query = compute_result(query_dataloader, net)

      # 转换成one-hot encoding，方便后续高效计算
      labels_database_one_hot = categorical_to_onehot(labels_database, class_num)
      labels_query_one_hot = categorical_to_onehot(labels_query, class_num)

      # (1) MAP, *****默认给所有得到的average precisions排序，top_k = -1*****
      MAP = compute_MAP(binaries_database.cpu().numpy(), binaries_query.cpu().numpy(), 
                        labels_database_one_hot.numpy(), labels_query_one_hot.numpy(), -1)

      # 根据自定义的labeling策略，得到预测的labels
      labels_pred_KNN = get_labels_pred_KNN(binaries_database.cpu().numpy(), binaries_query.cpu().numpy(), 
                       labels_database_one_hot.numpy(), labels_query_one_hot.numpy(), top_k)
      labels_pred_CHC = get_labels_pred_closest_hash_center(binaries_query.cpu().numpy(), labels_query.numpy(),
                                                            net.hash_centers.numpy())
      
      # (2) 自定义的labeling策略的accuracy
      labeling_accuracy_KNN = compute_labeling_strategy_accuracy(labels_pred_KNN, labels_query_one_hot.numpy())
      labeling_accuracy_CHC = compute_labeling_strategy_accuracy(labels_pred_CHC, labels_query.numpy())
      
      # (3) F1_score, average = (micro, macro, weighted)
      F1_score_weighted_average_KNN = f1_score(labels_query_one_hot, labels_pred_KNN, average='weighted')
      F1_score_per_class_KNN = f1_score(labels_query_one_hot, labels_pred_KNN, average=None)

      F1_score_weighted_average_CHC = f1_score(labels_query, labels_pred_CHC, average='weighted')
      F1_score_per_class_CHC = f1_score(labels_query, labels_pred_CHC, average=None)

      # (4) F1_score median
      F1_score_per_class_median_KNN = statistics.median(F1_score_per_class_KNN)
      F1_score_per_class_median_CHC = statistics.median(F1_score_per_class_CHC)

      KNN_metrics = (labeling_accuracy_KNN, F1_score_weighted_average_KNN, F1_score_per_class_median_KNN, F1_score_per_class_KNN)
      CHC_metrics = (labeling_accuracy_CHC, F1_score_weighted_average_CHC, F1_score_per_class_median_CHC, F1_score_per_class_CHC)

      return MAP, KNN_metrics, CHC_metrics
      


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


# top_k is hyper-parameter
# Predict label using KNN strategy (a)
def get_labels_pred_KNN(retrieval_binaries, query_binaries, retrieval_labels, query_labels, top_k):
    num_query = query_labels.shape[0]
    labels_pred = []
    for iter in range(num_query):
        hamm_dists = CalcHammingDist(query_binaries[iter, :], retrieval_binaries)

        hamm_indexes = np.argsort(hamm_dists)

        retrieval_labels_sort = retrieval_labels[hamm_indexes]

        topK_retrieval_sorted_labels = retrieval_labels_sort[0:top_k]

        # print("topK_retrieval_sorted_labels: ", topK_retrieval_sorted_labels)

        most_frequent_label = find_most_common_label(topK_retrieval_sorted_labels)

        labels_pred.append(most_frequent_label)

    return labels_pred


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
