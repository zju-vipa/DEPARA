import glob
import os 
import random
import argparse

parser = argparse.ArgumentParser(description='split dataset')

parser.add_argument('--part', dest='part', type=str)
parser.set_defaults(part='synthetic')

parser.add_argument('--data-dir', dest='data_dir', type=str)
parser.set_defaults(data_dir='data')

parser.add_argument('--split-path', dest='split_path', type=str)
parser.set_defaults(split_path='syn2real-data')

args = parser.parse_args()

prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if args.part == 'synthetic':
    data_origin = os.path.join(prj_dir, args.data_dir, 'train')
    file_extension = '.png'
elif args.part == 'mscoco':
    data_origin = os.path.join(prj_dir, args.data_dir, 'val')
    file_extension = '.jpg'
else:
    raise IOError('No such part in dataset syn2real')


split_dir = os.path.join(prj_dir, args.split_path)
if not os.path.exists(split_dir):
    os.mkdir(split_dir)

category = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']

if not os.path.exists(os.path.join(split_dir, args.part)):
    os.mkdir(os.path.join(split_dir, args.part))

if not os.path.exists(os.path.join(split_dir, args.part, 'train')):
    os.mkdir(os.path.join(split_dir, args.part, 'train'))

if not os.path.exists(os.path.join(split_dir, args.part, 'val')):
    os.mkdir(os.path.join(split_dir, args.part, 'val'))

if not os.path.exists(os.path.join(split_dir, args.part, 'test')):
    os.mkdir(os.path.join(split_dir, args.part, 'test'))

train_all, val_all, test_all = 0, 0, 0
for i in range(len(category)):
    imglist = glob.glob(os.path.join(data_origin, category[i], '*'+file_extension))
    random.shuffle(imglist)

    img_num = len(imglist)
    train_num = int(img_num * 0.7)
    val_num = int(img_num * 0.2)
    test_num = img_num - (train_num + val_num)
    print(category[i], ' : ', img_num)
    train_all += train_num
    val_all += val_num
    test_all += test_num

    if i < 9:
        if not os.path.exists(os.path.join(split_dir, args.part, 'train', '000'+str(i+1))):
            os.mkdir(os.path.join(split_dir, args.part, 'train', '000'+str(i+1)))
        if not os.path.exists(os.path.join(split_dir, args.part, 'val', '000'+str(i+1))):
            os.mkdir(os.path.join(split_dir, args.part, 'val', '000'+str(i+1)))
        for j in range(train_num):
            os.system('cp {} {}'.format(imglist[j], os.path.join(split_dir, args.part, 'train', '000'+str(i+1),
                                                                 os.path.basename(imglist[j]))))
        for j in range(val_num):
            os.system('cp {} {}'.format(imglist[j+train_num], os.path.join(split_dir, args.part, 'val', '000'+str(i+1),
                                                                           os.path.basename(imglist[j+train_num]))))
        for j in range(test_num):
            os.system('cp {} {}'.format(imglist[j+train_num+val_num], os.path.join(split_dir, args.part, 'test',
                                                                                   os.path.basename(imglist[j+train_num+val_num]))))
    else:
        if not os.path.exists(os.path.join(split_dir, args.part, 'train', '00'+str(i+1))):
            os.mkdir(os.path.join(split_dir, args.part, 'train', '00'+str(i+1)))
        if not os.path.exists(os.path.join(split_dir, args.part, 'val', '00'+str(i+1))):
            os.mkdir(os.path.join(split_dir, args.part, 'val', '00'+str(i+1)))
        for j in range(train_num):
            os.system('cp {} {}'.format(imglist[j], os.path.join(split_dir, args.part, 'train', '00'+str(i+1),
                                                                 os.path.basename(imglist[j]))))
        for j in range(val_num):
            os.system('cp {} {}'.format(imglist[j+train_num], os.path.join(split_dir, args.part, 'val', '00'+str(i+1),
                                                                           os.path.basename(imglist[j+train_num]))))
        for j in range(test_num):
            os.system('cp {} {}'.format(imglist[j+train_num+val_num], os.path.join(split_dir, args.part, 'test',
                                                                                   os.path.basename(imglist[j+train_num+val_num]))))

print('train all:', train_all)
print('val all:', val_all)
print('test all:', test_all)
