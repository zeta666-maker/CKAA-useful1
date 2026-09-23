from PIL import Image
import numpy as np
import pickle
import os
from tqdm import trange
from os.path import join


def my_mkdirs(path):

    if not os.path.exists(path):
        os.makedirs(path)


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')

    return dict


src_dir = 'A_CLData/cifar-100/cifar-100-python' 
dst_dir = 'A_CLData/cifar100-split' 


if __name__ == '__main__':
    meta = unpickle(join(src_dir, 'meta'))
    my_mkdirs(dst_dir)

    for data_set in ['train', 'test']:
        if data_set == 'train':
            data_set_name = 'train'
        else:
            data_set_name = 'val'
        print('Unpickling {} dataset......'.format(data_set))
        data_dict = unpickle(join(src_dir, data_set))
        my_mkdirs(join(dst_dir, data_set_name))

        for fine_label_name in data_dict['fine_labels']: ## 0~99
            my_mkdirs(join(dst_dir, data_set_name, str(fine_label_name))) 
        
        data = np.reshape(data_dict['data'], (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
        for i in range(data.shape[0]):
            # img = np.reshape(data_dict['data'][i], (3, 32, 32)).transpose(1,2,0)
            img = Image.fromarray(data[i])
            img.save(join(dst_dir, data_set_name, str(data_dict['fine_labels'][i]), str(i) + '.bmp'))

    print('All done.')