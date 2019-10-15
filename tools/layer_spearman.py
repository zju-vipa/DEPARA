# -*- coding: utf-8 -*-
import numpy as np
import os
import argparse
from scipy.spatial.distance import pdist, squareform


def cos_sim(vector_a, vector_b):
    """
    Cos Similarity
    :param vector_a: vector a
    :param vector_b: vector b
    :return: sim
    """
    return np.inner(vector_a, vector_b) / (np.linalg.norm(vector_a)*np.linalg.norm(vector_b))


def spearman(vi, vj):

    def rank(ind):
        l = ind.shape[0]
        r = np.zeros(l)
        for i in range(l):
            r[ind[i]] = i
        return r

    assert vi.shape == vj.shape
    ind_i = np.argsort(-vi)
    ind_j = np.argsort(-vj)
    rank_i = rank(ind_i)
    rank_j = rank(ind_j)
    s_corr = 1 - 6.0 * np.sum(np.square(rank_i - rank_j)) / (
                vi.shape[0] * (vi.shape[0] ** 2 - 1))
    return s_corr


def preprocess(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                matrix[i, j] = 0
    m = matrix.reshape(1, -1)
    m_ = m[m != 0]
    return m_.reshape(matrix.shape[0], matrix.shape[0]-1)


parser = argparse.ArgumentParser()

parser.add_argument('--dag-dir', dest='dag_dir', type=str)
parser.set_defaults(dag_dir='dag')

parser.add_argument('--source', dest='source', type=str)
parser.set_defaults(source='imagenet')

args = parser.parse_args()

if args.source == 'source':
    list_of_layer = 'conv_0 conv_3 conv_7 conv_10 \
    conv_14 conv_17 conv_20 conv_23 \
    conv_27 conv_30 conv_33 conv_36 \
    conv_40 conv_43 conv_46 \
    conv_49 linear_1 linear_4 linear_6'.split()
elif args.source == 'imagenet':
    list_of_layer_imagenet = 'conv_0 conv_3 conv_7 conv_10 \
    conv_14 conv_17 conv_20 conv_23 \
    conv_27 conv_30 conv_33 conv_36 \
    conv_40 conv_43 conv_46 \
    conv_49 linear_0 linear_3 linear_6'.split()
else:
    raise IOError('No such source')

target_layer = 'linear_target'

prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dag_dir = os.path.join(prj_dir, args.dag_dir)

'''edge'''
target_feature = np.load(os.path.join(dag_dir, 'feature_{}'.format(args.source), 'feature_{}.npy'.format(target_layer)))
target_feature = target_feature.reshape((target_feature.shape[0], -1))
target_edge = 1 - pdist(target_feature, 'cosine')

edge_list_vec = np.zeros((len(list_of_layer), 19900))
print('Edge:')
for i, layer in enumerate(list_of_layer):

    f = np.load(os.path.join(dag_dir, 'feature_{}'.format(args.source), 'feature_{}.npy'.format(layer)))

    f = f.reshape((f.shape[0], -1))
    edge_list_vec[i] = 1 - pdist(f, 'cosine')
    print(layer, ':', spearman(target_edge, edge_list_vec[i]))


'''node'''
target_attribution = np.load(os.path.join(dag_dir, 'attributionMap_{}'.format(args.source), 'attribution_{}.npy'.format(target_layer)))
target_attribution = np.abs(target_attribution).mean(axis=1).reshape((target_attribution.shape[0], -1))

print('Node:')
for i, layer in enumerate(list_of_layer):
    a = np.load(os.path.join(dag_dir, 'attributionMap_{}'.format(args.source), 'attribution_{}.npy'.format(layer)))
    a = np.abs(a).mean(axis=1).reshape((a.shape[0], -1))
    #a = a.mean(axis=1).reshape((a.shape[0], -1))
    sim = 0
    for k in range(a.shape[0]):
        cos = np.inner(a[k], target_attribution[k]) / (np.linalg.norm(a[k]) * np.linalg.norm(target_attribution[k]))
        sim += cos
    print(layer, sim/200)
