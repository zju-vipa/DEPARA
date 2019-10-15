# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np
import torch.optim as optim
import os
import time
import argparse
import json

from vgg import *
from torchvision import transforms
from torchvision import models


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
    parser = argparse.ArgumentParser(description='Pytorch Vgg transfer')

    # save and load
    parser.add_argument('--data-dir', dest='data_dir', type=str)
    parser.set_defaults(data_dir='syn2real-data-tenth')

    parser.add_argument('--model-weight-path', dest='model_weight_path', type=str)
    parser.set_defaults(model_weight_path='model_weights_syn2real/synthetic')

    parser.add_argument('--log-dir', dest='log_dir', type=str)
    parser.set_defaults(log_dir='log_save')

    parser.add_argument('--transfer-result-dir', dest='transfer_result_dir', type=str)
    parser.set_defaults(model_save_dir='transfer_result_layertrans')

    # network
    parser.add_argument('--source', dest='source', type=str)
    parser.set_defaults(source='synthetic')

    parser.add_argument('--net-arc', dest='net_arc', type=str)
    parser.set_defaults(net_arc='vgg19_bn')

    # training
    parser.add_argument('--lr', dest='lr', type=float)
    parser.set_defaults(lr=0.01)

    parser.add_argument('--batch-size', dest='batch_size', type=int)
    parser.set_defaults(batch_size=128)

    parser.add_argument('--epoch', dest='epoch', type=int)
    parser.set_defaults(epoch=50)

    parser.add_argument('--weight-decay', dest='weight_decay', type=float)
    parser.set_defaults(weight_decay=5e-4)

    parser.add_argument('--lr-step', dest='lr_step', type=str)
    parser.set_defaults(lr_step='[60, 120, 160]')

    parser.add_argument('--lr-drop-ratio', dest='lr_drop_ratio', type=float)
    parser.set_defaults(lr_drop_ratio=0.1)

    parser.add_argument('--gpu', dest='gpu', type=str)
    parser.set_defaults(gpu='0')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    epoch_step = json.loads(args.lr_step)
    data_path = os.path.join(prj_dir, args.data_dir)
    transfer_result_dir = os.path.join(prj_dir, args.transfer_result_dir)
    log_dir = os.path.join(prj_dir, args.log_dir)

    if not os.path.exists(transfer_result_dir):
        os.mkdir(transfer_result_dir)
    transfer_result_dir_source = os.path.join(transfer_result_dir, args.source)
    if not os.path.exists(transfer_result_dir_source):
        os.mkdir(transfer_result_dir_source)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    batch_size_ = args.batch_size
    data_class = 12
    data_name = 'mscoco'
    total_epoch = args.epoch
    avg_cost = np.zeros([total_epoch, 18, 4], dtype=np.float32)  # :[epoch, layer, 4]
    print('Fintuned using {}'.format(args.data_dir))

    im_train_set = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_path, data_name, 'train'),
                                               transform=data_transform(data_name, train=True)),
                                               batch_size=batch_size_,
                                               shuffle=True,
                                               num_workers=4, pin_memory=True)
    im_test_set = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_path, data_name, 'val'),
                                              transform=data_transform(data_name, train=False)),
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=4, pin_memory=True)

    # define vgg model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.source == 'syn':
        Vgg = vgg19_bn(num_class=data_class).to(device)
        numbered_layer = {
            'features': ['0', '3', '7', '10', '14', '17', '20', '23', '27', '30', '33', '36', '40', '43', '46', '49'],
            'classifier': ['1', '4']}
        list_of_layer = 'conv_0 conv_3 conv_7 conv_10 \
            conv_14 conv_17 conv_20 conv_23 \
            conv_27 conv_30 conv_33 conv_36 \
            conv_40 conv_43 conv_46 \
            conv_49 linear_1 linear_4'.split()

    elif args.source == 'Imagenet':
        Vgg = models.vgg19(pretrained=True).to(device)
        numbered_layer = {
            'features': ['0', '3', '7', '10', '14', '17', '20', '23', '27', '30', '33', '36', '40', '43', '46', '49'],
            'classifier': ['0', '3']}
        list_of_layer = 'conv_0 conv_3 conv_7 conv_10 \
                    conv_14 conv_17 conv_20 conv_23 \
                    conv_27 conv_30 conv_33 conv_36 \
                    conv_40 conv_43 conv_46 \
                    conv_49 linear_0 linear_3'.split()
    else:
        raise IOError('No such source')

    named_parameter_list = []
    for k in Vgg.named_parameters():
        named_parameter_list.append(k[0])

    for i in range(len(numbered_layer['features'])+len(numbered_layer['classifier'])):
        para_optim = []
        count_record = 0
        if i < len(numbered_layer['features']):
            submodule = 'features'
            ii = i
        else:
            submodule = 'classifier'
            ii = i - len(numbered_layer['features'])
        for k in Vgg.named_parameters():
            count_record += 1
            if k[0] == submodule + '.' + numbered_layer[submodule][ii] + '.bias':
                break

        count_iter = 0
        for k in Vgg.parameters():
            count_iter += 1
            if count_iter > count_record:
                para_optim.append(k)
            else:
                k.requires_grad = False

        optimizer = optim.SGD(para_optim, lr=args.lr, weight_decay=args.weight_decay, nesterov=True, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, epoch_step, gamma=args.lr_drop_ratio)

        if not os.path.exists(os.path.join(transfer_result_dir, data_name)):
            os.mkdir(os.path.join(transfer_result_dir, data_name))

        # load src's model params
        model_weight_path = os.path.join(prj_dir, args.model_weight_path)
        if args.model_weight_path is not None:
            pretrained_dict = torch.load(os.path.join(model_weight_path,
                                                      'vgg19_bn_final.pt'))
            model_dict = Vgg.state_dict()
            pretrained_dict_frozen = {}
            for k_, v_ in pretrained_dict.items():
                pretrained_dict_frozen[k_] = v_
                if k_ == submodule + '.' + numbered_layer[submodule][ii] + '.bias':
                    break

            print(pretrained_dict_frozen.keys())
            model_dict.update(pretrained_dict_frozen)
            Vgg.load_state_dict(model_dict)
            print('[*] Loaded {} ==> {}'.format(list_of_layer[i], data_name))

        print('Start Transfer TARGET:{} from Pretrained-MODEL-Layer:{}'.format(data_name, list_of_layer[i]))
        time_onelayer = time.time()
        for index in range(total_epoch):
            scheduler.step()
            time1 = time.time()

            cost = np.zeros(2, dtype=np.float32)
            train_dataset = iter(im_train_set)
            train_batch = len(train_dataset)
            Vgg.train()
            for _ in range(train_batch):
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
                avg_cost[index, i, 0:2] += cost / train_batch

            # evaluating test data
            Vgg.eval()
            test_dataset = iter(im_test_set)
            test_batch = len(test_dataset)
            for _ in range(test_batch):
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
                avg_cost[index, i, 2:] += cost / test_batch

            print('EPOCH: {:04d} | DATASET: {:s} Finetuned from {:s} || '
                  'TRAIN: {:.4f} {:.4f} || TEST: {:.4f} {:.4f} '
                  'TIME: {:.2f} minutes {:.2f} seconds'
                  .format(index, data_name, list_of_layer[i], avg_cost[index, i, 0], avg_cost[index, i, 1],
                          avg_cost[index, i, 2], avg_cost[index, i, 3], (time.time()-time1)//60, (time.time()-time1)%60))
            print('='*100)

            if not os.path.exists(os.path.join(transfer_result_dir_source, data_name, list_of_layer[i])):
                os.mkdir(os.path.join(transfer_result_dir_source, data_name, list_of_layer[i]))

            if index % 5 == 0:
                torch.save(Vgg.state_dict(), os.path.join(transfer_result_dir_source, data_name, list_of_layer[i], 'vgg19_bn_{}.pt'.format(index)))

        torch.save(Vgg.state_dict(), os.path.join(transfer_result_dir_source, data_name, list_of_layer[i], 'vgg19_bn_final.pt'))

        print('DATASET: {:s} Finetuned from {:s} : Time consumed: {:.2f} minutes {:.2f} seconds'.format(data_name, list_of_layer[i], (time.time()-time_onelayer)//60, (time.time()-time_onelayer)%60))
        print('cost:', avg_cost[:, i, :])
        np.save(os.path.join(transfer_result_dir_source, data_name, list_of_layer[i], 'cost_fc.npy'), avg_cost[:, i, :])

    np.save(os.path.join(transfer_result_dir_source, 'vgg19_bn_layertrans_{}.npy'.format(args.source)), avg_cost)


if __name__ == '__main__':
    main()
