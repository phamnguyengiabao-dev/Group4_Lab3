import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(y_true, y_pred):
    """Tính ACC sử dụng thuật toán Hungarian Matching"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def evaluate_clustering(y_true, y_pred):
    """Trả về 3 độ đo bắt buộc: ACC, NMI, ARI"""
    acc = cluster_accuracy(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    return acc, nmi, ari