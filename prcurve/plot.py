import numpy as np
import os
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import scipy.io
from scipy.special import comb, perm

parser = argparse.ArgumentParser(description='PR Curve')

parser.add_argument('--lamda', dest='lamda', type=float)
parser.set_defaults(lamda=0.25)

parser.add_argument('--lamda-coco', dest='lamda_coco', type=float)
parser.set_defaults(lamda_coco=0.03)

parser.add_argument('--lamda-indoor', dest='lamda_indoor', type=float)
parser.set_defaults(lamda_indoor=0.27)

parser.add_argument('--style', dest='style', type=str)
parser.set_defaults(style='seaborn')

parser.add_argument('--dataset', dest='dataset', type=str)
parser.set_defaults(dataset='taskonomy')

parser.add_argument('--save-dir', dest='save_dir', type=str)
parser.set_defaults(save_dir='result_save')

args = parser.parse_args()


def baseline():
    all_task = 19
    rel = 5
    nonrel = all_task - rel
    precision = []
    recall = []
    for fetch in range(1, 20):
        sum = 0
        for i in range(1, 6):
            sum += comb(rel, i) * comb(nonrel, fetch - i) * (i / fetch)
        sum /= comb(all_task, fetch)
        recall_one = sum * fetch / 5.0
        precision.append(sum)
        recall.append(recall_one)
    return precision, recall

def pr(gt_matrix, test_matrix):
  k = test_matrix.shape[1]
  num_intersect = 0
  for i in range(test_matrix.shape[0]):
      array_gt = gt_matrix[i].squeeze()
      array_test = test_matrix[i].squeeze()
      num_intersect += len(np.intersect1d(array_gt, array_test))
  precision = num_intersect / k / 18
  recall = num_intersect / 5 / 18
  return precision, recall


def pr_list(affinity, affinity_gt_rel):
  p_list, r_list = [], []
  ind_sort = np.argsort(-affinity, axis=1)
  for k in range(1, 20):
      test_matrix = ind_sort[:, 1:k+1]
      precision, recall = pr(affinity_gt_rel, test_matrix)
      p_list.append(precision)
      r_list.append(recall)
  precision = np.array(p_list).reshape(1,-1)
  recall = np.array(r_list).reshape(1,-1)
  p_r = np.concatenate((precision, recall), axis=0)
  return p_r


def preprocess(matrix):
  # delete 'Colorization' and 'In-painting' (not target)
  mat = np.delete(matrix, (7,19), axis=1)
  return mat

explain_methods = {'saliency':'saliency', 'grad*input':'gradXinput', 'elrp':'elrp'}
method_index_mapping = {'saliency': 0, 'grad*input': 1, 'elrp': 2}

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'
task_list = list_of_tasks.split(' ')

prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
result_save = os.path.join(prj_dir, args.save_dir)

affinity_gt = np.load('./sort_gt.npy')

# node
affinity = np.load(os.path.join(result_save, 'attribution', 'affinity_taskonomy.npy'))
affinity_coco = np.load(os.path.join(result_save, 'attribution', 'affinity_coco.npy'))
affinity_indoor = np.load(os.path.join(result_save, 'attribution', 'affinity_indoor.npy'))

# edge
affinity_edge = np.load(os.path.join(result_save, 'edge', 'edge_spearman_taskonomy.npy'))
affinity_edge_coco = np.load(os.path.join(result_save, 'edge', 'edge_spearman_coco.npy'))
affinity_edge_indoor = np.load(os.path.join(result_save, 'edge', 'edge_spearman_indoor.npy'))

# rsa
rsa_affinity = np.load(os.path.join(result_save, 'rsa', 'rsa_taskonomy.npy'))
rsa_coco = np.load(os.path.join(result_save, 'rsa', 'rsa_coco.npy'))
rsa_indoor = np.load(os.path.join(result_save, 'rsa', 'rsa_indoor.npy'))

# preprocess delete row 7 & 19 
affinity = preprocess(affinity)
affinity_coco = preprocess(affinity_coco)
affinity_indoor = preprocess(affinity_indoor)

rsa_coco = np.delete(rsa_coco, (7,19), axis=0)
rsa_indoor = np.delete(rsa_indoor, (7,19), axis=0)
rsa_affinity = np.delete(rsa_affinity, (7,19), axis=0)
affinity_edge = np.delete(affinity_edge, (7,19), axis=0)
affinity_edge_coco = np.delete(affinity_edge_coco, (7,19), axis=0)
affinity_edge_indoor = np.delete(affinity_edge_indoor, (7,19), axis=0)

aff_dict = {'taskonomy': affinity, 'coco': affinity_coco, 'indoor': affinity_indoor}
pr_dict = {}

# Get pr numerically
affinity_gt_rel = affinity_gt[:, 1:6]
for dataset_k, aff_v in aff_dict.items():
    for method, ind in method_index_mapping.items():
        affinity_oneMethod = aff_v[ind]
        pr_dict['{}_{}'.format(dataset_k, method)] = pr_list(affinity_oneMethod, affinity_gt_rel)

# get rsa
pr_dict['rsa_taskonomy'] = pr_list(rsa_affinity, affinity_gt_rel)
pr_dict['rsa_coco'] = pr_list(rsa_coco, affinity_gt_rel)
pr_dict['rsa_indoor'] = pr_list(rsa_indoor, affinity_gt_rel)

# get edgea
pr_dict['edge_taskonomy'] = pr_list(affinity_edge, affinity_gt_rel)
pr_dict['edge_coco'] = pr_list(affinity_edge_coco, affinity_gt_rel)
pr_dict['edge_indoor'] = pr_list(affinity_edge_indoor, affinity_gt_rel)

# get edge + node
affinity_edge_node = affinity[1]/1000. + args.lamda * affinity_edge
affinity_edge_node_coco = affinity_coco[1]/1000. + args.lamda_coco * affinity_edge_coco
affinity_edge_node_indoor = affinity_indoor[1]/1000. + args.lamda_indoor * affinity_edge_indoor

scipy.io.savemat('./affinity_depara.mat', {'affinity_depara': affinity_edge_node, 
                                           'affinity_depara_coco': affinity_edge_node_coco,
                                           'affinity_depara_indoor': affinity_edge_node_indoor})
pr_dict['edge+node_taskonomy'] = pr_list(affinity_edge_node, affinity_gt_rel)
pr_dict['edge+node_coco'] = pr_list(affinity_edge_node_coco, affinity_gt_rel)
pr_dict['edge+node_indoor'] = pr_list(affinity_edge_node_indoor, affinity_gt_rel)

# get rsa + node
# affinity_node_rsa = args.lamda * affinity[1]/1000. + rsa_affinity
# pr_dict['rsa+node'] = pr_list(affinity_node_rsa, affinity_gt_rel)

# get baseline
precision_base, recall_base = baseline()
x_axis = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19".split()

# get oracle
p_list, r_list = [], []
for k in range(1, 20):
    test_matrix_o = affinity_gt[:, 1:k+1]
    precision_oracle, recall_oracle = pr(affinity_gt_rel, test_matrix_o)
    p_list.append(precision_oracle)
    r_list.append(recall_oracle)
precision_oracle = np.array(p_list).reshape(1,-1)
recall_oracle = np.array(r_list).reshape(1,-1)
p_r = np.concatenate((precision_oracle, recall_oracle), axis=0)
pr_dict['oracle'] = p_r


plt.style.use(args.style)
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(15, 13))
plt.tick_params(labelsize=25)
# plot precision-k Curve
lines_p = plt.plot(x_axis, pr_dict['{}_grad*input'.format(args.dataset)][0],
                   x_axis, pr_dict['edge_{}'.format(args.dataset)][0],
                   x_axis, pr_dict['edge+node_{}'.format(args.dataset)][0],
                   x_axis, pr_dict['rsa_{}'.format(args.dataset)][0],
                   x_axis, precision_base,
                   x_axis, pr_dict['oracle'][0])


plt.setp(lines_p[0], color='lightcoral', linewidth=2, linestyle='-', marker='o', markersize=12, mec='lightcoral')
plt.setp(lines_p[1], color='green', linewidth=2, linestyle='-', marker='o', markersize=12, mec='green')
plt.setp(lines_p[2], color='gold', linewidth=2, linestyle='-', marker='o', markersize=12, mec='gold')
plt.setp(lines_p[3], color='blue', linewidth=2, linestyle='-', marker='o', markersize=12, mec='blue')
plt.setp(lines_p[4], color='black', linewidth=2, linestyle='-', marker='o', markersize=12, mec='black')
plt.setp(lines_p[5], color='red', linewidth=2, linestyle='-', marker='o', markersize=12, mec='red')


#线的标签
plt.legend(('DEPARA-$\mathcal{V}$',
            'DEPARA-$\mathcal{E}$',
            'DEPARA',
            'RSA',
            'Random ranking',
            'Oracle',), loc='best', prop={'size': 28})
plt.title('P@K Curve', {'size': 40})
plt.xlabel('K', {'size': 40})
plt.ylabel('Precision', {'size': 40})
plt.savefig('./{}_Precision-K-Curve.pdf'.format(args.dataset), dpi=1200)

plt.figure(figsize=(15, 13))
plt.tick_params(labelsize=25)
# plot precision-k Curve
lines_r = plt.plot(x_axis, pr_dict['{}_grad*input'.format(args.dataset)][1],
                   x_axis, pr_dict['edge_{}'.format(args.dataset)][1],
                   x_axis, pr_dict['edge+node_{}'.format(args.dataset)][1],
                   x_axis, pr_dict['rsa_{}'.format(args.dataset)][1],
                   x_axis, recall_base,
                   x_axis, pr_dict['oracle'][1])

plt.setp(lines_r[0], color='lightcoral', linewidth=2, linestyle='-', marker='o', markersize=12, mec='lightcoral')
plt.setp(lines_r[1], color='green', linewidth=2, linestyle='-', marker='o', markersize=12, mec='green')
plt.setp(lines_r[2], color='gold', linewidth=2, linestyle='-', marker='o', markersize=12, mec='gold')
plt.setp(lines_r[3], color='blue', linewidth=2, linestyle='-', marker='o', markersize=12, mec='blue')
plt.setp(lines_r[4], color='black', linewidth=2, linestyle='-', marker='o', markersize=12, mec='black')
plt.setp(lines_r[5], color='red', linewidth=2, linestyle='-', marker='o', markersize=12, mec='red')

#线的标签
plt.legend(('DEPARA-$\mathcal{V}$',
            'DEPARA-$\mathcal{E}$',
            'DEPARA',
            'RSA',
            'Random ranking',
            'Oracle',), loc='best', prop={'size': 28})
plt.title('R@K Curve', {'size': 40})
plt.xlabel('K', {'size': 40})
plt.ylabel('Recall', {'size': 40})
plt.savefig('./{}_Recall-K-Curve.pdf'.format(args.dataset), dpi=1200)



plt.figure(figsize=(15, 13))
plt.tick_params(labelsize=25)
# plot precision-recall Curve
lines_pr = plt.plot(pr_dict['{}_grad*input'.format(args.dataset)][1], pr_dict['{}_grad*input'.format(args.dataset)][0],
                    pr_dict['edge_{}'.format(args.dataset)][1], pr_dict['edge_{}'.format(args.dataset)][0],
                    pr_dict['edge+node_{}'.format(args.dataset)][1], pr_dict['edge+node_{}'.format(args.dataset)][0],
                    pr_dict['rsa_{}'.format(args.dataset)][1], pr_dict['rsa_{}'.format(args.dataset)][0],
                    recall_base, precision_base,
                    pr_dict['oracle'][1], pr_dict['oracle'][0])

plt.setp(lines_pr[0], color='lightcoral', linewidth=2, linestyle='-', marker='o', markersize=12, mec='lightcoral')
plt.setp(lines_pr[1], color='green', linewidth=2, linestyle='-', marker='o', markersize=12, mec='green')
plt.setp(lines_pr[2], color='gold', linewidth=2, linestyle='-', marker='o', markersize=12, mec='gold')
plt.setp(lines_pr[3], color='blue', linewidth=2, linestyle='-', marker='o', markersize=12, mec='blue')
plt.setp(lines_pr[4], color='black', linewidth=2, linestyle='-', marker='o', markersize=12, mec='black')
plt.setp(lines_pr[5], color='red', linewidth=2, linestyle='-', marker='o', markersize=12, mec='red')


# legend
plt.legend(('DEPARA-$\mathcal{V}$',
            'DEPARA-$\mathcal{E}$',
            'DEPARA',
            'RSA',
            'Random ranking',
            'Oracle',), loc='best', prop={'size': 28})

plt.title('PR Curve', {'size': 35})
plt.xlabel('Recall', {'size': 35})
plt.ylabel('Precision', {'size': 35})
plt.savefig('./{}_Precision-Recall-Curve.pdf'.format(args.dataset), dpi=1200)

