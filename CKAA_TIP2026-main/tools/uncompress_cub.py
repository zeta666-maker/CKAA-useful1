import os
import glob

dataset_dir = 'A_CLData/cub'

for dataset_name in ['train0', 'val0']:
    class_dirs = os.listdir(os.path.join(dataset_dir, dataset_name))
    if dataset_name == 'train0':
        dataset_name_ = 'train'
    else:
        dataset_name_ = 'val'

    with open('specific-shared/tools/cub200_' + str(dataset_name_) + '.txt', 'w') as file:
        file.truncate()
        for (class_id, class_dir) in enumerate(class_dirs):
            image_paths = os.listdir(os.path.join(dataset_dir, dataset_name, class_dir))
            for p in image_paths:
                file.write(os.path.join(dataset_name, class_dir, p) + ' ' + str(class_id) + '\n')