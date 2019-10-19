# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--feature-dir', dest='feature_dir', type=str)
parser.set_defaults(feature_dir='feature_dir')

parser.add_argument('--result-save', dest='result_save', type=str)
parser.set_defaults(result_save='result_save')

parser.add_argument('--source', dest='source', type=str)
parser.set_defaults(source='syn')

args = parser.parse_args()

prj_dir = os.path.dirname(os.path.dirname(os.path.basename(__file__)))
feature_dir = os.path.join(prj_dir, args.feature_dir)
result_save = os.path.join(prj_dir, args.result_save)
'''
list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'
list_of_tasks = list_of_tasks.split()
'''

if args.source == 'syn':
    list_of_layer = 'edge_conv_0 edge_conv_3 edge_conv_7 edge_conv_10 \
                     edge_conv_14 edge_conv_17 edge_conv_20 edge_conv_23 \
                     edge_conv_27 edge_conv_30 edge_conv_33 edge_conv_36 \
                     edge_conv_40 edge_conv_43 edge_conv_46 \
                     edge_conv_49 edge_linear_1 edge_linear_4 edge_linear_6 edge_target'.split()
    save_dir = 'edge_viz_syn'
elif args.source == 'imagenet':
    list_of_layer_imagenet = 'edge_conv_0 edge_conv_3 edge_conv_7 edge_conv_10 \
                              edge_conv_14 edge_conv_17 edge_conv_20 edge_conv_23 \
                              edge_conv_27 edge_conv_30 edge_conv_33 edge_conv_36 \
                              edge_conv_40 edge_conv_43 edge_conv_46 \
                              edge_conv_49 edge_linear_0 edge_linear_3 edge_linear_6 edge_target'.split()
    save_dir = 'edge_viz_imagenet'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

random.seed(1)
ind = np.arange(120)
random.shuffle(ind)
for k, layer in enumerate(list_of_layer):
    edge = np.load(os.path.join(feature_dir, 'edge_{}'.format(args.source), '{}.npy'.format(layer)))

    for i in range(edge.shape[0]):
        edge[i, i] = 1.
    edge = edge[ind, ...]
    edge = edge[..., ind]
    G = nx.Graph()
    G.clear()
    plt.clf()

    node_num = 30
    for i in range(node_num):
        for j in range(node_num):
            if i <= j:
                continue
            G.add_edge('{}'.format(i), '{}'.format(j), weight=edge[i, j])

    elarge = []
    ularge = []
    vlarge = []
    dlarge = []
    count = 0
    random.seed(2)
    threshold = 0.5
    '''
    if task == 'autoencoder':
        threshold = 0.47
    if task == 'curvature':
        threshold = 0.5
    if task == 'denoise':
        threshold = 0.5
    if task == 'edge2d':
        threshold = 0.5
    if task == 'edge3d':
        threshold = 0.5
    if task == 'keypoint2d':
        threshold = 0.905
    if task == 'keypoint3d':
        threshold = 0.5
    if task == 'colorization':
        threshold = 0.5
    if task == 'reshade':
        threshold = 0.5
    if task == 'rgb2depth':
        threshold = 0.48
    if task == 'rgb2mist':
        threshold = 0.48
    if task == 'rgb2sfnorm':
        threshold = 0.5
    if task == 'room_layout':
        threshold = 0.8865
    if task == 'segment25d':
        threshold = 0.5
    if task == 'segment2d':
        threshold = 0.68
    if task == 'vanishing_point':
        threshold = 0.7755
    if task == 'segmentsemantic':
        threshold = 0.5
    if task == 'class_1000':
        threshold = 0.885
    if task == 'class_places':
        threshold = 0.9
    if task == 'inpainting_whole':
        threshold = 0.5
    '''
    # SOURCE
    if args.source == 'syn':
        if layer == 'edge_conv_0':
            threshold = 0.2
        if layer == 'edge_conv_3':
            threshold = 0.5
        if layer == 'edge_conv_7':
            threshold = 0.45
        if layer == 'edge_conv_10':
            threshold = 0.38
        if layer == 'edge_conv_14':
            threshold = 0.42
        if layer == 'edge_conv_17':
            threshold = 0.39
        if layer == 'edge_conv_20':
            threshold = 0.37
        if layer == 'edge_conv_23':
            threshold = 0.352
        if layer == 'edge_conv_27':
            threshold = 0.45
        if layer == 'edge_conv_30':
            threshold = 0.31
        if layer == 'edge_conv_33':
            threshold = 0.36
        if layer == 'edge_conv_36':
            threshold = 0.34
        if layer == 'edge_conv_40':
            threshold = 0.445
        if layer == 'edge_conv_43':
            threshold = 0.41
        if layer == 'edge_conv_46':
            threshold = 0.32
        if layer == 'edge_conv_49':
            threshold = 0.15
        if layer == 'edge_linear_1':
            threshold = 0.3
        if layer == 'edge_linear_4':
            threshold = 0.3
        if layer == 'edge_linear_6':
            threshold = 0.3
        if layer == 'edge_target':
            threshold = 0.1
    elif args.source == 'imagenet':
        if layer == 'edge_conv_0':
            threshold = 0.01
        if layer == 'edge_conv_3':
            threshold = 0.72
        if layer == 'edge_conv_7':
            threshold = 0.6
        if layer == 'edge_conv_10':
            threshold = 0.53
        if layer == 'edge_conv_14':
            threshold = 0.44
        if layer == 'edge_conv_17':
            threshold = 0.57
        if layer == 'edge_conv_20':
            threshold = 0.65
        if layer == 'edge_conv_23':
            threshold = 0.615
        if layer == 'edge_conv_27':
            threshold = 0.45
        if layer == 'edge_conv_30':
            threshold = 0.57
        if layer == 'edge_conv_33':
            threshold = 0.665
        if layer == 'edge_conv_36':
            threshold = 0.75
        if layer == 'edge_conv_40':
            threshold = 0.32
        if layer == 'edge_conv_43':
            threshold = 0.6
        if layer == 'edge_conv_46':
            threshold = 0.64
        if layer == 'edge_conv_49':
            threshold = 0.4
        if layer == 'edge_linear_0':
            threshold = 0.2
        if layer == 'edge_linear_3':
            threshold = 0.2
        if layer == 'edge_linear_6':
            threshold = 0.25

    for (u, v, d) in G.edges(data=True):
        count += 1
        prob = random.random()
        # print(u,v,d)
        if d['weight'] > threshold and prob < 0.05:
            # count += 1
            elarge.append((u, v, d['weight']))
            # ularge.append(u)
            # vlarge.append(v)
            # dlarge.append(d['weight'])

    '''
    sel = int(count * 0.03)  # 21
    ind = np.argsort(-np.array(dlarge))
    if sel > len(ind):
        sel = len(ind)
    for i in range(sel):
        elarge.append((ularge[int(ind[i])], vlarge[int(ind[i])]))
    '''
    # position
    # pos = nx.spring_layout(G)  # positions for all nodes
    pos = nx.circular_layout(G)
    # pos = nx.shell_layout(G)
    # pos = nx.fruchterman_reingold_layout(G)
    # pos = nx.random_layout(G)

    # nodes
    colors = np.arange(node_num)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=colors, alpha=0.8)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge,
                           width=2, alpha=1, edge_color='c')
    #nx.draw_networkx_edges(G, pos, edgelist=esmall,
    #                       width=0.4, alpha=0.5, edge_color='m', style='dashed')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=15, font_family='sans-serif')

    plt.axis('off')
    plt.tight_layout()
    #plt.show()
    #plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0)
    plt.savefig(os.path.join(result_save, save_dir, '{}_edge_viz.pdf'.format(layer)), dpi=1200)
