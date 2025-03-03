import os

# path = '../../data/'
# if not os.path.isdir(path):
#     os.mkdir(path)

# train segmentation
txtName = 'HAM10000_seg.txt'
f = open(txtName, 'a+')

path = "../../data/HAM10000/"
path_mask = "../../data/HAM10000_segmentations/"

path_list = os.listdir(path)
path_list_mask = os.listdir(path_mask)
path_list.sort()
path_list_mask.sort()

for i in range(len(path_list)):
    trainIMG = path[-16::]+path_list[i]
    trainGT = path_mask[-29::]+path_list_mask[i]
    result = trainIMG + ' ' + trainGT +'\n'
    f.write(result)

f.close()

# # val segmentation
# txtName = 'ISIC/Validation_seg.txt'
# f = open(txtName, 'a+')

# path = "seg_data/ISIC-2017_Validation_Data/Images/"
# path_mask = "seg_data/ISIC-2017_Validation_Data/Annotation/"

# path_list = os.listdir(path)
# path_list_mask = os.listdir(path_mask)
# path_list.sort()
# path_list_mask.sort()

# for i in range(len(path_list)):
#     trainIMG = path[-7::]+path_list[i]
#     trainGT = path_mask[-11::]+path_list_mask[i]
#     result = trainIMG + ' ' + trainGT +'\n'
#     f.write(result)

# f.close()

# # test segmentation
# txtName = 'ISIC/Testing_seg.txt'
# f = open(txtName, 'a+')

# path = "seg_data/ISIC-2017_Testing_Data/Images/"
# path_mask = "seg_data/ISIC-2017_Testing_Data/Annotation/"

# path_list = os.listdir(path)
# path_list_mask = os.listdir(path_mask)
# path_list.sort()
# path_list_mask.sort()

# for i in range(len(path_list)):
#     trainIMG = path[-7::]+path_list[i]
#     trainGT = path_mask[-11::]+path_list_mask[i]
#     result = trainIMG + ' ' + trainGT +'\n'
#     f.write(result)

# f.close()