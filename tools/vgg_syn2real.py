# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np
import torch.optim as optim
import os
import time
import argparse

from vgg import *
from torchvision.transforms import transforms


def data_transform(name, train=True):

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    if name in ['mscoco']:  # no horz flip
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    elif name in ['synthetic']:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if name in ['mscoco']:  # no horz flip
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    elif name in ['synthetic']:
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if train:
        return transform_train
    else:
        return transform_test


# main()
def main():
    parser = argparse.ArgumentParser(description='Pytorch VGG Syn2real Training Baseline')

    # save and load
    parser.add_argument('--data-dir', dest='data_dir', type=str)
    parser.set_defaults(data_dir='syn2real-data')

    parser.add_argument('--model-weight-path', dest='model_weight_path', type=str)
    parser.set_defaults(model_weight_path=None)

    parser.add_argument('--log-dir', dest='log_dir', type=str)
    parser.set_defaults(log_dir='log_save_syn2real')

    parser.add_argument('--model-save-dir', dest='model_save_dir', type=str)
    parser.set_defaults(model_save_dir='model_weights_syn2real')

    # network
    parser.add_argument('--net-arc', dest='net_arc', type=str)
    parser.set_defaults(net_arc='vgg19_bn')

    # training
    parser.add_argument('--task-name', dest='task_name', type=str)
    parser.set_defaults(task_name='mscoco')

    parser.add_argument('--lr', dest='lr', type=float)
    parser.set_defaults(lr=0.01)

    parser.add_argument('--batch-size', dest='batch_size', type=int)
    parser.set_defaults(batch_size=32)

    parser.add_argument('--epoch', dest='epoch', type=int)
    parser.set_defaults(epoch=60)

    parser.add_argument('--weight-decay', dest='weight_decay', type=float)
    parser.set_defaults(weight_decay=5e-4)

    parser.add_argument('--lr-step', dest='lr_step', type=str)
    parser.set_defaults(lr_step='[60, 120, 160]')

    parser.add_argument('--lr-step-const', dest='lr_step_c', type=int)
    parser.set_defaults(lr_step_c=200)

    parser.add_argument('--lr-drop-ratio', dest='lr_drop_ratio', type=float)
    parser.set_defaults(lr_drop_ratio=0.1)

    parser.add_argument('--gpu', dest='gpu', type=str)
    parser.set_defaults(gpu='0')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_dir = os.path.join(prj_dir, args.log_dir)
    model_save_dir = os.path.join(prj_dir, args.model_save_dir)
    

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    data_path = os.path.join(prj_dir, args.data_dir)
    data_name = args.task_name
    data_class = 12

    total_epoch = args.epoch
    avg_cost = np.zeros([total_epoch, 4], dtype=np.float32)
    im_train_set = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_path, data_name, 'train'),
                                               transform=data_transform(data_name, train=True)),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4, pin_memory=True)
    im_test_set = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_path, data_name, 'val'),
                                              transform=data_transform(data_name, train=False)),
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=4, pin_memory=True)

    # define model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.net_arc == 'vgg19_bn':
        Vgg = vgg19_bn(num_class=data_class).to(device)
    elif args.net_arc == 'vgg16_bn':
        Vgg = vgg16_bn(num_class=data_class).to(device)
    else:
        raise Exception('vgg16 or vgg19')
    optimizer = optim.SGD(Vgg.parameters(), lr=args.lr, weight_decay=args.weight_decay, nesterov=True, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step_c, gamma=args.lr_drop_ratio)

    if args.model_weight_path is not None:
        model_weight_path = os.path.join(prj_dir, args.model_weight_path)
        Vgg.load_state_dict(torch.load(model_weight_path))
        print('[*]Model loaded : ({})'.format(os.path.basename(model_weight_path)))

    print('Training DATASET:{}'.format(data_name))
    time_onedataset = time.time()
    for index in range(total_epoch):
        scheduler.step()
        time1 = time.time()

        cost = np.zeros(2, dtype=np.float32)
        train_dataset = iter(im_train_set)
        train_batch = len(train_dataset)
        Vgg.train()
        for k in range(train_batch):
            train_data, train_label = train_dataset.next()
            train_label = train_label.type(torch.LongTensor)
            train_data, train_label = train_data.to(device), train_label.to(device)
            train_pred1 = Vgg(train_data)

            # reset optimizer with zero gradient
            optimizer.zero_grad()
            train_loss1 = Vgg.model_fit(train_pred1, train_label, device=device, num_output=data_class)
            train_loss = torch.mean(train_loss1)
            train_loss.backward()
            optimizer.step()

            # calculate training loss and accuracy
            train_predict_label1 = train_pred1.data.max(1)[1]
            train_acc1 = train_predict_label1.eq(train_label).sum().item() / train_data.shape[0]

            cost[0] = torch.mean(train_loss1).item()
            cost[1] = train_acc1
            avg_cost[index, :2] += cost / train_batch

        # evaluating test data
        Vgg.eval()
        test_dataset = iter(im_test_set)
        test_batch = len(test_dataset)
        with torch.no_grad():
            for k in range(test_batch):
                test_data, test_label = test_dataset.next()
                test_label = test_label.type(torch.LongTensor)
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_pred1 = Vgg(test_data)

                test_loss1 = Vgg.model_fit(test_pred1, test_label, device=device, num_output=data_class)

                # calculate testing loss and accuracy
                test_predict_label1 = test_pred1.data.max(1)[1]
                test_acc1 = test_predict_label1.eq(test_label).sum().item() / test_data.shape[0]

                cost[0] = torch.mean(test_loss1).item()
                cost[1] = test_acc1
                avg_cost[index, 2:] += cost / test_batch

        print('EPOCH: {:04d} | DATASET: {:s} || TRAIN: {:.4f} {:.4f} || TEST: {:.4f} {:.4f} TIME: {:.2f} minutes {:.2f} seconds'
              .format(index, data_name, avg_cost[index, 0], avg_cost[index, 1],
                      avg_cost[index, 2], avg_cost[index, 3], (time.time()-time1)//60, (time.time()-time1)%60))
        print('='*100)

        if not os.path.exists(os.path.join(model_save_dir, data_name)):
            os.mkdir(os.path.join(model_save_dir, data_name))

        if index % 5 == 0:
            torch.save(Vgg.state_dict(), os.path.join(model_save_dir, data_name,
                                                      '{}_{}.pt'.format(args.net_arc, index)))

    np.save(os.path.join(args.model_save_dir, data_name, 'cost.npy'), avg_cost)
    torch.save(Vgg.state_dict(), os.path.join(model_save_dir, data_name,
                                              '{}_final.pt'.format(args.net_arc)))
    print('DATASET: {:s} : Time consumed: {:.2f} hours {:.2f} minutes {:.2f} seconds'
          .format(data_name, (time.time()-time_onedataset)//3600, ((time.time()-time_onedataset)%3600)//60, (time.time()-time_onedataset)%60))

    np.save(os.path.join(log_dir, '{}_{}_train_log_baseline.npy'.format(args.net_arc, data_name)), avg_cost)


if __name__ == '__main__':
    main()

