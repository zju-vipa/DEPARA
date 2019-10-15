import numpy as np
import os
import glob
import random
import argparse

parser = argparse.ArgumentParser('Dataset Shrinkage')

parser.add_argument('--data-dir', dest='data_dir', type=str)
parser.set_defaults(data_dir='syn2real-data')

parser.add_argument('--save-dir', dest='save_dir', type=str)
parser.set_defaults(save_dir='syn2real-data-tenth')

parser.add_argument('--shrink-ratio', dest='shrink_ratio', type=int)
parser.set_defaults(shrink_ratio=10)

args = parser.parse_args()

data_name = ['mscoco']
prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
save_dir = os.path.join(prj_dir, args.save_dir)
if not os.path.exists(save_dir):
	os.mkdir(save_dir)

for i in range(len(data_name)):
	dataset = data_name[i]
	if not os.path.exists(os.path.join(save_dir + dataset)):
		os.mkdir(os.path.join(save_dir + dataset))
	sel_num = 0
	for mode in ['train']:
		cate_list = glob.glob('./{}/{}/*'.format(dataset, mode))
		for j in range(len(cate_list)):
			img_list = glob.glob(cate_list[j] + '/*.jpg')
			subfolder = cate_list[j].split('/', 3)[3]
			random.seed(i)
			random.shuffle(img_list)
			if not os.path.exists(os.path.join(save_dir, dataset, mode)):
				os.mkdir(os.path.join(save_dir, dataset, mode))
			if not os.path.exists(os.path.join(save_dir, dataset, mode, subfolder)):
				os.mkdir(os.path.join(save_dir, dataset, mode, subfolder))

			one_tenth_sel = len(img_list) // args.shrink_ratio
			sel_num += one_tenth_sel
			for k in range(one_tenth_sel):
				picName = save_dir + dataset + '/' + mode + '/' + subfolder + '/' + os.path.basename(img_list[k])
				os.system('cp {} {}'.format(img_list[k], picName))
	print('{} train:{}'.format(dataset, sel_num))
	os.system('cp -r {} {}'.format(os.path.join(prj_dir, 'syn2real-data', dataset, 'val'), os.path.join(save_dir, dataset)))
	os.system('cp -r {} {}'.format(os.path.join(prj_dir, 'syn2real-data', dataset, 'test'), os.path.join(save_dir, dataset)))
