import os
import scipy.io as scio
import json


dataset_dir = os.path.join('A_CLData', 'stanford_cars')
txt_dir = 'specific-shared/tools'
dataset_name = ['cars_train', 'cars_test']

# test_gt = scio.loadmat('A_CLData/stanford_cars/cars_test_annos_withlabels.mat')
# test_lbl = []
# txt_path = os.path.join(txt_dir, 'stanford_cars_test' + '.txt')
# with open(txt_path, 'w') as file:
#     file.truncate()
#     for gt in test_gt['annotations'][0]:
#         try:
#             image_path = os.path.join(dataset_dir, dataset_name[1], gt[-1][0])
#             image_lbl = gt[-2][0,0].astype(int)
#             line_to_write = os.path.join(dataset_name[1], gt[-1][0]) + ' ' + str(image_lbl) + '\n'
#             print("Writing:", line_to_write)  # Debugging output
#             file.write(line_to_write)
#         except Exception as e:
#             print("Error:", e)


# with open(f"A_CLData/stanford_cars/devkit/train_perfect_preds.txt") as f:
#     lines = f.readlines()

txt_path = os.path.join(txt_dir, 'stanford_cars_test' + '.txt')
json_file_path = 'A_CLData/stanford_cars/split_zhou_StanfordCars.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    train_data = json.load(file)

files = []
train_lbl = []
for data in train_data['test']:
    files.append(data[0])
    train_lbl.append(int(data[1]))
        
txt_path = os.path.join(txt_dir, 'stanford_cars_test' + '.txt')
with open(txt_path, 'w') as file:
    file.truncate()
    for (image_path, image_lbl) in zip(files, train_lbl):
        try:
            line_to_write = os.path.join(image_path) + ' ' + str(image_lbl) + '\n'
            print("Writing:", line_to_write)  # Debugging output
            file.write(line_to_write)
        except Exception as e:
            print("Error:", e)

files = []
train_lbl = []
for data in train_data['train']:
    files.append(data[0])
    train_lbl.append(int(data[1]))
for data in train_data['val']:
    files.append(data[0])
    train_lbl.append(int(data[1]))
        
txt_path = os.path.join(txt_dir, 'stanford_cars_train' + '.txt')
with open(txt_path, 'w') as file:
    file.truncate()
    for (image_path, image_lbl) in zip(files, train_lbl):
        try:
            line_to_write = os.path.join(image_path) + ' ' + str(image_lbl) + '\n'
            print("Writing:", line_to_write)  # Debugging output
            file.write(line_to_write)
        except Exception as e:
            print("Error:", e)