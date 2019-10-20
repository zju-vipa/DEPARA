# -*- coding: utf-8 -*-
from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--feature-dir', dest='feature_dir', type=str)
parser.set_defaults(feature_dir='feature')

parser.add_argument('--save-dir', dest='save_dir', type=str)
parser.set_defaults(save_dir='result_save')

parser.add_argument('--dataset', dest='dataset', type=str)
parser.set_defaults(dataset='coco')

args = parser.parse_args()


def spearman_correlation(matrix):
    spearman_corr = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i is j:
                spearman_corr[i, j] = 1
            elif i < j:
                continue
            else:
                def rank(ind):
                    l = ind.shape[0]
                    r = np.zeros(l)
                    for i in range(l):
                        r[ind[i]] = i
                    return r
                ind_i = np.argsort(-matrix[i])
                ind_j = np.argsort(-matrix[j])
                rank_i = rank(ind_i)
                rank_j = rank(ind_j)
                spearman_corr[i, j] = 1 - 6.0 * np.sum(np.square(rank_i-rank_j)) / (matrix.shape[1]*(matrix.shape[1]**2-1))
                spearman_corr[j, i] = spearman_corr[i, j]
    return spearman_corr


if __name__ == '__main__':
    list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
    keypoint2d keypoint3d colorization \
    reshade rgb2depth rgb2mist rgb2sfnorm \
    room_layout segment25d segment2d vanishing_point \
    segmentsemantic class_1000 class_places inpainting_whole'.split()

    prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    directory_save = os.path.join(prj_dir, args.feature_dir, '{}_feature_1k'.format(args.dataset))
    result_save = os.path.join(prj_dir, args.save_dir)

    if not os.path.exists(result_save):
        os.mkdir(result_save)
    if not os.path.exists(os.path.join(result_save, 'edge')):
        os.mkdir(os.path.join(result_save, 'edge'))
    if not os.path.exists(os.path.join(result_save, 'rsa')):
        os.mkdir(os.path.join(result_save, 'rsa'))
       
    feature = np.load(os.path.join(directory_save, list_of_tasks[0], 'task_feature.npy'))
    feature_all = np.zeros((20, feature.shape[0]*(feature.shape[0]-1)//2))
    feature_all_correlation = np.zeros((20, feature.shape[0]*(feature.shape[0]-1)//2))
    for i, task in enumerate(list_of_tasks):
        feature = np.load(os.path.join(directory_save, task, 'task_feature.npy'))
        feature = feature - np.mean(feature, axis=0)
        feature_cosine = pdist(feature, 'cosine')
        feature_all[i] = feature_cosine
        feature_correlation = pdist(feature, 'correlation')
        feature_all_correlation[i] = feature_correlation

    spearman_20x20 = spearman_correlation(feature_all)
    spearman_20x20_correlation = spearman_correlation(feature_all_correlation)
    np.save(os.path.join(result_save, 'edge', 'edge_spearman_{}.npy'.format(args.dataset)), spearman_20x20)
    np.save(os.path.join(result_save, 'rsa', 'rsa_{}.npy'.format(args.dataset)), spearman_20x20_correlation)


