# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import os
import argparse

from utils import *
from vgg import *
from model_wrn_mtan import *
from vanilla_backprop import VanillaBackprop
from torchvision import models
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import pdist, squareform

parser = argparse.ArgumentParser(description='Pytorch VGG Layer Feature Extract')

parser.add_argument('--target', action='store_true')
parser.set_defaults(target=False)

parser.add_argument('--model-weight-path', dest='model_weight_path', type=str)
parser.set_defaults(model_weight_path='model_weights_syn2real')

parser.add_argument('--data-dir', dest='data_dir', type=str)
parser.set_defaults(data_dir='syn2real-data')

parser.add_argument('--save-dir', dest='save_dir', type=str)
parser.set_defaults(save_dir='dag')

parser.add_argument('--imlist', dest='imlist', type=str)
parser.set_defaults(imlist='imlist_200.txt')

parser.add_argument('--net-arc', dest='net_arc', type=str)
parser.set_defaults(net_arc='vgg19_bn')

parser.add_argument('--gpu', dest='gpu', type=str)
parser.set_defaults(gpu='1')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
save_dir = os.path.join(prj_dir, args.save_dir)
data_dir = os.path.join(prj_dir, args.data_dir)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

class FeatureExtractor(nn.Module):

    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        for name, module in self.submodule._modules.items():
            if name is 'features':
                self.feature = module
            if name is 'classifier':
                self.classifier = module
        if 'feature' in self.extracted_layers.keys():
            self.feature_layerlist = self.extracted_layers['feature']
        if 'classifier' in self.extracted_layers.keys():
            self.classifier_layerlist = self.extracted_layers['classifier']

    def forward(self, x):
        outputs = []

        for name, module in self.feature._modules.items():
            x = module(x)
            if 'feature' in self.extracted_layers.keys() and name in self.feature_layerlist:
                outputs.append(x.data.cpu().numpy())

        x = x.view(x.size(0), -1)
        for name, module in self.classifier._modules.items():
            x = module(x)
            if 'classifier' in self.extracted_layers.keys() and name in self.classifier_layerlist:
                outputs.append(x.data.cpu().numpy())

        return outputs


def data_transform(img):

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])

    return t(img).unsqueeze_(0)


def generate_attribution_map(img_batch, model, numbered_layer, device):
    VBP = VanillaBackprop(model)
    myExtractor = FeatureExtractor(model, extracted_layers=numbered_layer)
    output = myExtractor(img_batch)

    grad = []
    if 'feature' in numbered_layer.keys():
        for i in range(len(numbered_layer['feature'])):
            grad.append(VBP.generate_gradients(img_batch, numbered_layer['feature'][i], 'feature', device) * img_batch.data.cpu().numpy())

    if 'classifier' in numbered_layer.keys():
        for i in range(len(numbered_layer['classifier'])):
            grad.append(VBP.generate_gradients(img_batch, numbered_layer['classifier'][i], 'classifier', device) * img_batch.data.cpu().numpy())

    return (output, grad)


def save(output, grad, edge, pretrained, numbered_layer):
    if not os.path.exists(os.path.join(save_dir, 'feature_{}'.format(pretrained))):
        os.mkdir(os.path.join(save_dir, 'feature_{}'.format(pretrained)))
    if not os.path.exists(os.path.join(save_dir, 'node_{}'.format(pretrained))):
        os.mkdir(os.path.join(save_dir, 'node_{}'.format(pretrained)))
    if not os.path.exists(os.path.join(save_dir, 'edge_{}'.format(pretrained))):
        os.mkdir(os.path.join(save_dir, 'edge_{}'.format(pretrained)))

    for ind in range(len(numbered_layer['feature'])):
        np.save(os.path.join(save_dir, 'feature_{}'.format(pretrained),
                             'feature_conv_{}.npy'.format(numbered_layer['feature'][ind])), output[ind])
        np.save(os.path.join(save_dir, 'node_{}'.format(pretrained),
                             'node_conv_{}.npy'.format(numbered_layer['feature'][ind])), grad[ind])
        np.save(os.path.join(save_dir, 'edge_{}'.format(pretrained),
                             'edge_conv_{}.npy'.format(numbered_layer['feature'][ind])), edge[ind])

    for ind in range(len(numbered_layer['classifier'])):
        np.save(os.path.join(save_dir, 'feature_{}'.format(pretrained),
                             'feature_linear_{}.npy'.format(numbered_layer['classifier'][ind])),
                              output[len(numbered_layer['feature']) + ind])
        np.save(os.path.join(save_dir, 'attributionMap_{}'.format(pretrained),
                             'attribution_linear_{}.npy'.format(numbered_layer['classifier'][ind])),
                              grad[len(numbered_layer['feature']) + ind])
        np.save(os.path.join(save_dir, 'edge_{}'.format(pretrained),
                             'edge_linear_{}.npy'.format(numbered_layer['classifier'][ind])),
                              edge[len(numbered_layer['feature']) + ind])


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # input
    path_list = os.path.join(data_dir, args.imlist)
    img_list = []
    with open(path_list, 'r') as f:
        for _ in range(200):
            img_list += f.readline().split()

    imgs = None
    for i in range(200):
        print(i, ' image')
        img = Image.open(os.path.join(data_dir, 'test200', img_list[i])).convert('RGB')
        if imgs is None:
            imgs = data_transform(img)
        else:
            imgs = torch.cat((imgs, data_transform(img)))
    imgs = imgs.to(device)
    imgs = torch.autograd.Variable(imgs, requires_grad=True)

    print('Images Loaded')

    # load model
    Vgg = vgg19_bn(num_class=12).to(device)
    Vgg.load_state_dict(torch.load(os.path.join(args.model_weight_path, 'synthetic', 'vgg19_bn_final.pt')))
    Vgg.eval()

    Vgg_imagenet = models.vgg19_bn(pretrained=True).to(device)
    Vgg_imagenet.eval()

    if args.target:
        numbered_layer = {'classifier': ['6']}
        numbered_layer_imagenet = {'classifier': ['6']}
        list_length = len(numbered_layer['classifier'])
        target_layer = ['linear_target']
    else:
        numbered_layer = {
            'feature': ['0', '3', '7', '10', '14', '17', '20', '23', '27', '30', '33', '36', '40', '43', '46', '49'],
            'classifier': ['1', '4', '6']}
        numbered_layer_imagenet = {
            'feature': ['0', '3', '7', '10', '14', '17', '20', '23', '27', '30', '33', '36', '40', '43', '46', '49'],
            'classifier': ['0', '3', '6']}
        list_length = len(numbered_layer['feature']) + len(numbered_layer['classifier'])

    batch_size = 20
    for i in range(len(img_list)//batch_size):
        print('Iter:', i+1)
        img_batch = torch.autograd.Variable(imgs[i * batch_size: (i + 1) * batch_size], requires_grad=True)
        (output, grad) = generate_attribution_map(img_batch, Vgg, numbered_layer, device=device)
        (output_imagenet, grad_imagenet) = generate_attribution_map(img_batch, Vgg_imagenet, numbered_layer_imagenet, device=device)
        if i == 0:
            output_list = output
            grad_list = grad
            output_list_imagenet = output_imagenet
            grad_list_imagenet = grad_imagenet
            continue
        for j in range(len(img_list)//batch_size-1):
            output_list[j] = np.concatenate((output_list[j], output[j]), axis=0)
            grad_list[j] = np.concatenate((grad_list[j], grad[j]), axis=0)
            output_list_imagenet[j] = np.concatenate((output_list_imagenet[j], output_imagenet[j]), axis=0)
            grad_list_imagenet[j] = np.concatenate((grad_list_imagenet[j], grad_imagenet[j]), axis=0)

    edge_list = []
    edge_list_imagenet = []
    for i in range(list_length):
        f, f_imagenet = output_list[i], output_list_imagenet[i]
        f, f_imagenet = f.reshape((f.shape[0], -1)), f_imagenet.reshape((f_imagenet.shape[0], -1))
        edge_list.append(squareform(1 - pdist(f, 'cosine')))
        edge_list_imagenet.append(squareform(1 - pdist(f_imagenet, 'cosine')))

    if args.target:
        numbered_layer['classifier'] = target_layer
        numbered_layer_imagenet['classifier'] = target_layer
    save(output_list, grad_list, edge_list, 'source', numbered_layer)
    save(output_list_imagenet, grad_list_imagenet, edge_list_imagenet, 'imagenet', numbered_layer_imagenet)




