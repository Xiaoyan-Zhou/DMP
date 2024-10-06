import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from data_utils import DatasetLoader
from torch.utils.data import DataLoader
import time
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import mode

def get_device(device_number):
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    if use_cuda and device_number == '1':
        device = torch.device("cuda:1" if use_cuda else "cpu")
        print(device)
    elif use_cuda and device_number == '0':
        device = torch.device("cuda:0" if use_cuda else "cpu")
        print(device)
    return device

if __name__ == '__main__':
    #total number of clustering
    K = 70
    #path of train data
    data_path = r'E:\1-PHD\01ZXY\dataset_paper\1-OODdata\MSTAR\SOC'
    model_path = './init_model_npy/resnet18CE.pt'
    centroids_path = f'./init_model_npy/resnet18CE_10_K{K}_HierarchicalClustering_centroids_fixed.npy'
    labels_path = f'./init_model_npy/resnet18CE_10_K{K}_HierarchicalClustering_labels_fixed.npy'

    model = torch.load(model_path)
    model_f = torch.nn.Sequential(*list(model.children())[:-1])
    device = get_device('0')
    model_f = model_f.to(device)
    model = model.to(device)
    model_f = model_f.to(device)

    trainset = DatasetLoader('train', data_path)
    train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        with tqdm(train_loader, total=len(train_loader)) as pbar:
            end = time.time()
            feature_list = []
            label_true = []
            for idx, (images, label) in enumerate(pbar):
                if torch.cuda.is_available():
                    images = images.to(device)
                features = model_f(images)
                feature_list.append(features.flatten().cpu()) # append flatten features
                label_true.append(label.flatten())

    features_array = np.array(feature_list)
    label_array = np.array(label_true)
    # 进行层次聚类
    # Z = linkage(features_array, method='ward')  # 使用Ward方法
    Z = linkage(features_array, method='single')  # 使用single方法

    # 根据设定的聚类中心数量切割树状图，形成聚类
    clusters = fcluster(Z, t=K, criterion='maxclust')

    centroids = np.array([features_array[clusters == k].mean(axis=0) for k in range(1, K + 1)])
    llabel = np.array([label_array[clusters == k].mean(axis=0) for k in range(1, K + 1)])
    print(centroids.shape, llabel.shape)
    print('llabel', llabel)
    # print('lpath', lpath)
    np.save(centroids_path, centroids)
    np.save(labels_path, llabel)
####https://www.cnblogs.com/jin-liang/p/9527522.html 利用散点图展示聚类效果
    from sklearn.cluster import AgglomerativeClustering

    # create clusters
    # hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    # save clusters for chart
    # y_hc = clusters.fit_predict(features_array)
    #
    # plt.scatter(features_array[y_hc == 0, 0], features_array[y_hc == 0, 1], s=100, c='red')
    # plt.scatter(features_array[y_hc == 1, 0], features_array[y_hc == 1, 1], s=100, c='black')
    # plt.scatter(features_array[y_hc == 2, 0], features_array[y_hc == 2, 1], s=100, c='blue')
    # plt.scatter(features_array[y_hc == 3, 0], features_array[y_hc == 3, 1], s=100, c='cyan')
