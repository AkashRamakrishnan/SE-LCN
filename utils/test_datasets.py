import numpy as np
from datasets import train_segmentation_data, val_segmentation_data, train_symmetry_data, val_symmetry_data, train_classification_data, val_classification_data

root_path = '/home/s3075451/'
# data_dir = '../../data/SymDerm_v2/SymDerm_extended/'
data_dir = '../../data/HAM10000_inpaint/'
train_data = 'data_files/HAM10000_seg_train.txt'
val_data = 'data_files/HAM10000_seg_val.txt'
train_csv = 'data_files/HAM10000_metadata_train.csv'
val_csv = 'data_files/HAM10000_metadata_val.csv'

# train_set = train_segmentation_data(root_path, train_data)
train_set = train_classification_data(data_dir, train_csv)

image, label, name = train_set[0]

print("image")
print(image.shape)
print(image)

print("label")
print(label)

print("name")
print(name)

# val_set = val_segmentation_data(root_path, val_data)
val_set = val_classification_data(data_dir, val_csv)

val_image, val_label, val_name = val_set[0]

print("val image")
print(val_image.shape)
print(val_image)

print("val_label")
print(val_label)

print('val name')
print(val_name)