import os
data_dir = 'A_CLData/imagenet-a'
name = ['train', 'val']

name = 'train'
mode_dir = os.path.join(data_dir, name)
mode_dir_dirs = os.listdir(mode_dir)
with open('specific-shared/tools/imagenet_a_classnames.txt', 'w') as f:
    for (i, dir) in enumerate(mode_dir_dirs):
        f.write(dir + '\t' + str(i) + '\n')
        
with open('specific-shared/tools/imagenet_a_train.txt', 'w') as f:
    for (i, dir) in enumerate(mode_dir_dirs):        
        files = os.listdir(os.path.join(data_dir, name, dir))
        for file in files:
            file = file.replace(" ", "")
            f.write(str(os.path.join(dir, file) + '\n'))

name = 'val'
with open('specific-shared/tools/imagenet_a_val.txt', 'w') as f:
    for (i, dir) in enumerate(mode_dir_dirs):        
        files = os.listdir(os.path.join(data_dir, name, dir))
        for file in files:
            file = file.replace(" ", "")
            f.write(str(os.path.join(dir, file) + '\n'))