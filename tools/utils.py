# -*- coding: utf-8 -*-
import torch
import pickle
import os
import numpy as np
import random

from torchvision.transforms import transforms


def data_transform(data_path, name, train=True):
    with open(os.path.join(data_path, 'decathlon_mean_std_plus.pickle'), 'rb') as handle:
        dict_mean_std = pickle._Unpickler(handle)
        dict_mean_std.encoding = 'latin1'
        dict_mean_std = dict_mean_std.load()

    means = dict_mean_std[name + 'mean']
    stds = dict_mean_std[name + 'std']

    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_train = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(72),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if train:
        return transform_train
    else:
        return transform_test


def regress_loss(train_regress, feature_gt):
    return torch.norm(train_regress - feature_gt, dim=1)


def get_img_label(im_set):
    dataset = iter(im_set)
    batches = len(dataset)
    img, label = None, None
    for i in range(batches):
        img_, label_ = dataset.next()
        if img is None:
            img = img_
            label = label_
        else:
            img = torch.cat([img, img_], 0)
            label = torch.cat([label, label_], 0)
    return img, label


def mix_dataset(im_set_target, im_set_source):
    img_t, label_t = get_img_label(im_set_target)
    target = np.ones(img_t.shape[0])
    img_s, label_s = get_img_label(im_set_source)
    source = np.zeros(img_s.shape[0])
    # concatenate
    img = torch.cat([img_t, img_s], 0)
    indicator = np.concatenate((target, source), axis=0)
    label = torch.cat([label_t, label_s], 0)
    # shuffle
    random.seed(10)
    idx = np.arange(img_t.size(0)+img_s.size(0))
    random.shuffle(idx)
    img = img[idx]
    indicator = indicator[idx]
    label = label[idx]
    return (img, indicator, label), img_t.size(0), img_s.size(0)


def detach_from_mix(batch):
    (img_batch, indicator_batch, label_batch) = batch
    img_batch_s, img_batch_t, label_batch_s, label_batch_t = None, None, None, None

    for j in range(img_batch.size(0)):
        if indicator_batch[j] == 1:
            if img_batch_t is None:
                img_batch_t = torch.unsqueeze(img_batch[j], dim=0)
                label_batch_t = label_batch[j].view(1, -1)
            else:
                img_batch_t = torch.cat([img_batch_t, torch.unsqueeze(img_batch[j], dim=0)], dim=0)
                label_batch_t = torch.cat([label_batch_t, label_batch[j].view(1, -1)], dim=1)

        if indicator_batch[j] == 0:
            if img_batch_s is None:
                img_batch_s = torch.unsqueeze(img_batch[j], dim=0)
                label_batch_s = label_batch[j].view(1, -1)
            else:
                img_batch_s = torch.cat([img_batch_s, torch.unsqueeze(img_batch[j], dim=0)], dim=0)
                label_batch_s = torch.cat([label_batch_s, label_batch[j].view(1, -1)], dim=1)
    label_batch_t = label_batch_t.view(-1)
    label_batch_s = label_batch_s.view(-1)
    return (img_batch_t, label_batch_t), (img_batch_s, label_batch_s)


def reg(mat):
    return torch.div(mat, torch.norm(mat, dim=1).view(-1, 1))
