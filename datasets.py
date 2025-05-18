import numpy as np
import torchvision.transforms.functional as tf
import random
from torch.utils import data
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import torch

class train_segmentation_data(data.Dataset):
    def __init__(self, img_dir, mask_dir, train_df, crop_size=(224, 224), max_iters=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.train_df = train_df
        self.crop_w, self.crop_h = crop_size
        self.max_iters = max_iters
        
    
        # self.train_augmentation = transforms.Compose(
        #     [transforms.RandomVerticalFlip(p=0.5),
        #      transforms.RandomHorizontalFlip(p=0.5),
        #      transforms.ToTensor(),
        #      transforms.ToPILImage(),
        #      ])

        # self.train_gt_augmentation = transforms.Compose(
        #     [transforms.RandomVerticalFlip(p=0.5),
        #      transforms.RandomHorizontalFlip(p=0.5),
        #      transforms.ToTensor(),
        #      transforms.ToPILImage()
        #      ])
                
        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])

        self.train_gt_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])

    def __len__(self):
        return len(self.train_df)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.train_df['image_id'][idx]+'.png')
        label_path = os.path.join(self.mask_dir, self.train_df['image_id'][idx]+'_segmentation.png')
        image = Image.open(image_path)
        label = Image.open(label_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)
        name = image_path.split('/')[-1].split('.')[0]
        seed = np.random.randint(2147483647)
        random.seed(seed)
        image = self.train_augmentation(image)

        random.seed(seed)
        label = self.train_gt_augmentation(label)

        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)

        label = np.array(label)
        label = np.float32(label > 0)

        return image.copy(), label.copy(), name


class val_segmentation_data(data.Dataset):
    def __init__(self, img_dir, mask_dir,  val_df, crop_size=(224, 224), max_iters=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.val_df = val_df
        self.crop_w, self.crop_h = crop_size
        self.max_iters = max_iters


        self.val_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])

        self.val_gt_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])


    def __len__(self):
        return len(self.val_df)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.val_df['image_id'][idx]+'.png')
        label_path = os.path.join(self.mask_dir, self.val_df['image_id'][idx]+'_segmentation.png')
        image = Image.open(image_path)
        label = Image.open(label_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)
        name = image_path.split('/')[-1].split('.')[0]
        # seed = np.random.randint(2147483647)
        # random.seed(seed)
        image = self.val_augmentation(image)

        # random.seed(seed)
        label = self.val_gt_augmentation(label)

        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)

        label = np.array(label)
        label = np.float32(label > 0)

        return image.copy(), label.copy(), name

class test_segmentation_data(data.Dataset):
    def __init__(self, root_path, list_path, crop_size=(224, 224), max_iters=None):
        self.root_path = root_path
        self.list_path = list_path
        self.crop_w, self.crop_h = crop_size
        self.max_iters = max_iters
        self.files = []
        
        with open(self.list_path) as f:
            lines = f.readlines()
            for line in lines:
                image, mask = line.strip().split()
                self.files.append({
                "img": image,
                "label": mask})


        self.test_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])

        self.test_gt_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_path, self.files[idx]['img'])
        label_path = os.path.join(self.root_path, self.files[idx]['label'])
        image = Image.open(image_path)
        label = Image.open(label_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        label = label.resize((self.crop_w, self.crop_h), Image.NEAREST)
        name = image_path.split('/')[-1].split('.')[0]
        # seed = np.random.randint(2147483647)
        # random.seed(seed)
        image = self.test_augmentation(image)

        # random.seed(seed)
        label = self.test_gt_augmentation(label)

        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)

        label = np.array(label)
        label = np.float32(label > 0)

        return image.copy(), label.copy(), name
    
class run_segmentation_data(data.Dataset):
    def __init__(self, root_path, crop_size=(224, 224), max_iters=None):
        self.root_path = root_path
        self.files = os.listdir(root_path)
        self.crop_w, self.crop_h = crop_size
        self.max_iters = max_iters
        

        self.test_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])

        self.test_gt_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_path, self.files[idx])
        image = Image.open(image_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        name = image_path.split('/')[-1].split('.')[0]
        # seed = np.random.randint(2147483647)
        # random.seed(seed)
        image = self.test_augmentation(image)

        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)

        return image.copy(), name
    
class run_eval_cam(data.Dataset):
    def __init__(self, root_path, mask_root_path, crop_size=(224, 224), max_iters=None, HAM=False):
        self.root_path = root_path
        self.mask_root_path = mask_root_path
        self.files = os.listdir(root_path)
        self.crop_w, self.crop_h = crop_size
        self.max_iters = max_iters
        self.HAM = HAM
        

        self.test_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])

        self.test_gt_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_path, self.files[idx])
        if self.HAM:
            mask_path = os.path.join(self.mask_root_path, self.files[idx].split('.')[0]+'_Segmentation.png')
        else:
            mask_path = os.path.join(self.mask_root_path, self.files[idx].split('.')[0]+'.png')
        image = Image.open(image_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        mask = Image.open(mask_path)
        mask = mask.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        name = image_path.split('/')[-1].split('.')[0]
        # seed = np.random.randint(2147483647)
        # random.seed(seed)
        image = self.test_augmentation(image)
        mask = self.test_gt_augmentation(mask)
        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        mask = np.array(mask)
        mask = (mask>0.5).astype(np.uint8)

        return image.copy(), mask.copy(), name
    

class run_eval_cam_final(data.Dataset):
    def __init__(self, root_path, mask_root_path, cam_root_path, crop_size=(224, 224), max_iters=None):
        self.root_path = root_path
        self.mask_root_path = mask_root_path
        self.cam_root_path = cam_root_path
        self.crop_w, self.crop_h = crop_size
        self.max_iters = max_iters

        # Get the list of image files
        image_files = set(os.listdir(root_path))
        
        # Get the list of mask files and remove the suffix '_segmentation.png' to match image filenames
        mask_files = set(f.replace('_segmentation.png', '.png') for f in os.listdir(mask_root_path))
        
        # Get the list of CAM files and remove the suffix '.npy' to match image filenames
        cam_files = set(f.replace('.npy', '.png') for f in os.listdir(cam_root_path))
        
        # Use intersection to ensure the filenames are present in all three directories
        # valid_files = image_files & mask_files & cam_files
        
        # print(len(image_files), len(mask_files), len(cam_files), len(valid_files))
        # print(mask_files)
        # Convert back to list and store
        # self.files = [f for f in valid_files]  # Adjust the extension as necessary
        self.files = os.listdir(root_path)  # Adjust the extension as necessary
        print(len(self.files))
        self.test_augmentation = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage()
        ])

        self.test_gt_augmentation = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage()
        ])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_path, self.files[idx])
        mask_path = os.path.join(self.mask_root_path, self.files[idx].split('.')[0] + '_Segmentation.png')
        cam_path = os.path.join(self.cam_root_path, self.files[idx].split('.')[0] + '.npy')
        
        image = Image.open(image_path).resize((self.crop_w, self.crop_h), Image.BICUBIC)
        mask = Image.open(mask_path).resize((self.crop_w, self.crop_h), Image.BICUBIC)
        
        name = self.files[idx].split('/')[-1].split('.')[0]
        image = self.test_augmentation(image)
        mask = self.test_gt_augmentation(mask)
        
        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        
        mask = np.array(mask)
        mask = (mask > 0.5).astype(np.uint8)

        cam = np.load(cam_path)

        return image.copy(), mask.copy(), cam.copy(), name

class train_symmetry_data(data.Dataset):
    def __init__(self, datadir, dataframe, crop_size = (224,224), mask = False):
        self.datadir = datadir
        self.crop_w, self.crop_h = crop_size
        self.data = dataframe
        self.mask = mask
        # self.train_augmentation = transforms.Compose(
        #     [transforms.RandomVerticalFlip(p=0.5),
        #      transforms.RandomHorizontalFlip(p=0.5),
        #      transforms.RandomRotation(90),
        #      transforms.ToTensor(),
        #      transforms.ToPILImage()
        #      ])
                
        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mask:
            image_path = os.path.join(self.datadir, self.data['Mask'][idx])
        else:
            image_path = os.path.join(self.datadir, self.data['File'][idx])
        image = Image.open(image_path)
        label = self.data['labels_symmetry'][idx]
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.train_augmentation(image)
        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image.copy(), label.copy(), name
    

class val_symmetry_data(data.Dataset):
    def __init__(self, datadir, dataframe, crop_size = (224,224), mask=False):
        self.datadir = datadir
        self.crop_w, self.crop_h = crop_size
        self.data = dataframe
        self.mask = mask

        self.val_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.mask:
            image_path = os.path.join(self.datadir, self.data['Mask'][idx])
        else:
            image_path = os.path.join(self.datadir, self.data['File'][idx])
        image = Image.open(image_path)
        label = self.data['labels_symmetry'][idx]
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.val_augmentation(image)
        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image, label, name


# class train_symmetry_data(data.Dataset):
#     def __init__(self, datadir, dataframe, crop_size = (224,224), mask = False):
#         self.datadir = datadir
#         self.crop_w, self.crop_h = crop_size
#         self.data = dataframe
#         self.mask = mask
#         # self.train_augmentation = transforms.Compose(
#         #     [transforms.RandomVerticalFlip(p=0.5),
#         #      transforms.RandomHorizontalFlip(p=0.5),
#         #      transforms.RandomRotation(90),
#         #      transforms.ToTensor(),
#         #      transforms.ToPILImage()
#         #      ])
                
#         self.train_augmentation = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.ToPILImage()
#              ])
            
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         if self.mask:
#             image_path = os.path.join(self.datadir, self.data['Mask'][idx])
#         else:
#             image_path = os.path.join(self.datadir, self.data['File'][idx])
#         image = Image.open(image_path)
#         label = self.data['labels_symmetry'][idx]
#         image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         image = self.train_augmentation(image)
#         image = np.array(image) / 255.
#         image = image.transpose((2, 0, 1))
#         image = image.astype(np.float32)
#         label = np.array(int(label))
#         name = image_path.split('/')[-1].split('.')[0]
#         return image.copy(), label.copy(), name
    

# class val_symmetry_data(data.Dataset):
#     def __init__(self, datadir, dataframe, crop_size = (224,224), mask=False):
#         self.datadir = datadir
#         self.crop_w, self.crop_h = crop_size
#         self.data = dataframe
#         self.mask = mask

#         self.val_augmentation = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.ToPILImage()
#              ])
            
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         if self.mask:
#             image_path = os.path.join(self.datadir, self.data['Mask'][idx])
#         else:
#             image_path = os.path.join(self.datadir, self.data['File'][idx])
#         image = Image.open(image_path)
#         label = self.data['labels_symmetry'][idx]
#         image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         image = self.val_augmentation(image)
#         image = np.array(image) / 255.
#         image = image.transpose((2, 0, 1))
#         image = image.astype(np.float32)
#         label = np.array(int(label))
#         name = image_path.split('/')[-1].split('.')[0]
#         return image, label, name
        
# class train_symmetry_data_4c(data.Dataset):
#     def __init__(self, datadir, maskdir, train_csv, crop_size = (224,224)):
#         self.datadir = datadir
#         self.maskdir = maskdir
#         self.crop_w, self.crop_h = crop_size
#         self.data = pd.read_csv(train_csv)
#         # self.train_augmentation = transforms.Compose(
#         #     [transforms.RandomVerticalFlip(p=0.5),
#         #      transforms.RandomHorizontalFlip(p=0.5),
#         #      transforms.ToTensor(),
#         #      transforms.ToPILImage()
#         #      ])
                
#         self.train_augmentation = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.ToPILImage()
#              ])
            
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         image_path = os.path.join(self.datadir, self.data['File'][idx])
#         image = Image.open(image_path)
#         mask_path = os.path.join(self.maskdir, self.data['Mask'][idx])
#         mask = Image.open(mask_path)
#         label = self.data['labels_symmetry'][idx]
#         image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         mask = mask.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         image = self.train_augmentation(image)
#         image = np.array(image) / 255.
#         mask = np.array(mask) 
#         mask= np.where(mask> 127, 255, 0)
#         mask = mask / 255.
#         mask = mask[:, :, np.newaxis]
#         image = np.concatenate((image, mask), axis=-1)
#         image = image.transpose((2, 0, 1))
#         image = image.astype(np.float32)
#         label = np.array(int(label))
#         name = image_path.split('/')[-1].split('.')[0]
#         return image.copy(), label.copy(), name
    

# class val_symmetry_data_4c(data.Dataset):
#     def __init__(self, datadir, maskdir, val_csv, crop_size = (224,224)):
#         self.datadir = datadir
#         self.crop_w, self.crop_h = crop_size
#         self.data = pd.read_csv(val_csv)
#         self.maskdir = maskdir
#         self.val_augmentation = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.ToPILImage()
#              ])
            
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         image_path = os.path.join(self.datadir, self.data['File'][idx])
#         image = Image.open(image_path)
#         mask_path = os.path.join(self.maskdir, self.data['Mask'][idx])
#         mask = Image.open(mask_path)
#         label = self.data['labels_symmetry'][idx]
#         image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         mask = mask.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         image = self.val_augmentation(image)
#         image = np.array(image) / 255.
#         mask = np.array(mask) 
#         mask= np.where(mask> 127, 255, 0)
#         mask = mask / 255.
#         mask = mask[:, :, np.newaxis]
#         image = np.concatenate((image, mask), axis=-1)
#         image = image.transpose((2, 0, 1))
#         image = image.astype(np.float32)
#         label = np.array(int(label))
#         name = image_path.split('/')[-1].split('.')[0]
#         return image.copy(), label.copy(), name


class train_classification_data(data.Dataset):
    def __init__(self, datadir, train_df, crop_size = (224,224)):
        self.datadir = datadir
        self.files = []
        self.crop_w, self.crop_h = crop_size
        self.LUT = {'nv': 0,
                    'mel': 1,
                    'bkl': 2, 
                    'bcc': 3, 
                    'akiec': 4, 
                    'vasc': 5, 
                    'df': 6}
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.data = train_df

        self.data['labels'] = self.data['dx'].map(self.LUT)
        self.data = self.data[['image_id', 'labels']]

        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
            #  transforms.Normalize(mean, std),
             transforms.ToPILImage()
             ])

            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data['image_id'][idx]
        label = self.data['labels'][idx]
        image_path = os.path.join(self.datadir, image_id+'.png')
        image = Image.open(image_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.train_augmentation(image)
        image = np.array(image) / 255.
        image = np.array(image) 
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image.copy(), label.copy(), name


class val_classification_data(data.Dataset):
    def __init__(self, datadir, val_df, crop_size = (224,224)):
        self.datadir = datadir
        self.files = []
        self.crop_w, self.crop_h = crop_size
        self.LUT = {'nv': 0,
                    'mel': 1,
                    'bkl': 2, 
                    'bcc': 3, 
                    'akiec': 4, 
                    'vasc': 5, 
                    'df': 6}
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.data = val_df

        self.data['labels'] = self.data['dx'].map(self.LUT)
        self.data = self.data[['image_id', 'labels']]

        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
            #  transforms.Normalize(mean, std), 
             transforms.ToPILImage()
             ])

            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data['image_id'][idx]
        label = self.data['labels'][idx]
        image_path_png = os.path.join(self.datadir, image_id + '.png')
        image_path_jpg = os.path.join(self.datadir, image_id + '.jpg')

        if os.path.exists(image_path_png):
            image_path = image_path_png
        elif os.path.exists(image_path_jpg):
            image_path = image_path_jpg
        else:
            raise FileNotFoundError(f"Image file for ID {image_id} not found with .png or .jpg extension.")
        image = Image.open(image_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.train_augmentation(image)
        image = np.array(image) / 255.
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image.copy(), label.copy(), name

class test_2019_data(data.Dataset):
    def __init__(self, datadir, val_df, crop_size = (224,224)):
        self.datadir = datadir
        self.files = []
        self.crop_w, self.crop_h = crop_size
        self.LUT = {'NV': 0,
                    'MEL': 1,
                    'BKL': 2, 
                    'BCC': 3, 
                    'AK': 4, 
                    'VASC': 5, 
                    'DF': 6}
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.data = val_df

        self.data['labels'] = self.data['category'].map(self.LUT)
        self.data = self.data[['image', 'labels']]

        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
            #  transforms.Normalize(mean, std), 
             transforms.ToPILImage()
             ])

            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data['image'][idx]
        label = self.data['labels'][idx]
        image_path = os.path.join(self.datadir, image_id+'.jpg')
        image = Image.open(image_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.train_augmentation(image)
        image = np.array(image) / 255.
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image.copy(), label.copy(), name

class test_2019_cam(data.Dataset):
    def __init__(self, datadir, camdir, val_df, crop_size = (224,224)):
        self.datadir = datadir
        self.camdir = camdir
        self.files = []
        self.crop_w, self.crop_h = crop_size
        self.LUT = {'NV': 0,
                    'MEL': 1,
                    'BKL': 2, 
                    'BCC': 3, 
                    'AK': 4, 
                    'VASC': 5, 
                    'DF': 6}
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.data = val_df

        self.data['labels'] = self.data['category'].map(self.LUT)
        self.data = self.data[['image', 'labels']]

        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
            #  transforms.Normalize(mean, std), 
             transforms.ToPILImage()
             ])

            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data['image'][idx]
        label = self.data['labels'][idx]
        image_path = os.path.join(self.datadir, image_id+'.jpg')
        cam_path = os.path.join(self.camdir, image_id+'.npy')
        cam = np.load(cam_path)
        image = Image.open(image_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.train_augmentation(image)
        image = np.array(image) / 255.
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image.copy(), label.copy(), cam.copy(), name

class train_classification_2016(data.Dataset):
    def __init__(self, datadir, train_df, crop_size = (224,224)):
        self.datadir = datadir
        self.files = []
        self.crop_w, self.crop_h = crop_size
        self.LUT = {'benign': 0,
                    'malignant': 1}
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.data = train_df

        self.data['labels'] = self.data['class'].map(self.LUT)
        self.data = self.data[['image_id', 'labels']]

        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
            #  transforms.Normalize(mean, std),
             transforms.ToPILImage()
             ])

            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data['image_id'][idx]
        label = self.data['labels'][idx]
        image_path = os.path.join(self.datadir, image_id +'.png')
        image = Image.open(image_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.train_augmentation(image)
        image = np.array(image) / 255.
        image = np.array(image) 
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image.copy(), label.copy(), name


class val_classification_2016(data.Dataset):
    def __init__(self, datadir, val_df, crop_size = (224,224)):
        self.datadir = datadir
        self.files = []
        self.crop_w, self.crop_h = crop_size
        self.LUT = {'benign': 0,
                    'malignant': 1}
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.data = val_df

        self.data['labels'] = self.data['class'].map(self.LUT)
        self.data = self.data[['image_id', 'labels']]

        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
            #  transforms.Normalize(mean, std), 
             transforms.ToPILImage()
             ])

            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data['image_id'][idx]
        label = self.data['labels'][idx]
        image_path = os.path.join(self.datadir, image_id+'.png')
        image = Image.open(image_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.train_augmentation(image)
        image = np.array(image) / 255.
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image.copy(), label.copy(), name

class train_classification_cam_2016(data.Dataset):
    def __init__(self, datadir, camdir, train_df, crop_size = (224,224)):
        self.datadir = datadir
        self.camdir = camdir
        self.files = []
        self.crop_w, self.crop_h = crop_size
        self.LUT = {'benign': 0,
                    'malignant': 1}
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.data = train_df

        self.data['labels'] = self.data['class'].map(self.LUT)
        self.data = self.data[['image_id', 'labels']]

        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
            #  transforms.Normalize(mean, std),
             transforms.ToPILImage()
             ])

            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data['image_id'][idx]
        label = self.data['labels'][idx]
        image_path = os.path.join(self.datadir, image_id+'.png')
        cam_path = os.path.join(self.camdir, image_id+'.npy')
        cam = np.load(cam_path)
        image = Image.open(image_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.train_augmentation(image)
        image = np.array(image) / 255.
        image = np.array(image) 
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image.copy(), label.copy(), cam.copy(), name


class val_classification_cam_2016(data.Dataset):
    def __init__(self, datadir, camdir, val_df, crop_size = (224,224)):
        self.datadir = datadir
        self.camdir = camdir
        self.files = []
        self.crop_w, self.crop_h = crop_size
        self.LUT = {'benign': 0,
                    'malignant': 1}
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.data = val_df

        self.data['labels'] = self.data['class'].map(self.LUT)
        self.data = self.data[['image_id', 'labels']]

        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
            #  transforms.Normalize(mean, std), 
             transforms.ToPILImage()
             ])

            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data['image_id'][idx]
        label = self.data['labels'][idx]
        image_path = os.path.join(self.datadir, image_id+'.png')
        cam_path = os.path.join(self.camdir, image_id+'.npy')
        image = Image.open(image_path)
        cam = np.load(cam_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.train_augmentation(image)
        image = np.array(image) / 255.
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image.copy(), label.copy(), cam.copy(), name

class train_classification_cam(data.Dataset):
    def __init__(self, datadir, camdir, train_df, crop_size = (224,224)):
        self.datadir = datadir
        self.camdir = camdir
        self.files = []
        self.crop_w, self.crop_h = crop_size
        self.LUT = {'nv': 0,
                    'mel': 1,
                    'bkl': 2, 
                    'bcc': 3, 
                    'akiec': 4, 
                    'vasc': 5, 
                    'df': 6}
        
        self.data = train_df

        self.data['labels'] = self.data['dx'].map(self.LUT)
        self.data = self.data[['image_id', 'labels']]

        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])

            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data['image_id'][idx]
        label = self.data['labels'][idx]
        image_path = os.path.join(self.datadir, image_id+'.png')
        cam_path = os.path.join(self.camdir, image_id+'.npy')
        cam = np.load(cam_path)
        image = Image.open(image_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.train_augmentation(image)
        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image.copy(), label.copy(), cam.copy(), name


class val_classification_cam(data.Dataset):
    def __init__(self, datadir, camdir, val_df, crop_size = (224,224)):
        self.datadir = datadir
        self.camdir = camdir
        self.files = []
        self.crop_w, self.crop_h = crop_size
        self.LUT = {'nv': 0,
                    'mel': 1,
                    'bkl': 2, 
                    'bcc': 3, 
                    'akiec': 4, 
                    'vasc': 5, 
                    'df': 6}
        
        self.data = val_df

        self.data['labels'] = self.data['dx'].map(self.LUT)
        self.data = self.data[['image_id', 'labels']]

        self.train_augmentation = transforms.Compose(
            [transforms.ToTensor(),
             transforms.ToPILImage()
             ])

            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data['image_id'][idx]
        label = self.data['labels'][idx]
        
        image_path_png = os.path.join(self.datadir, image_id + '.png')
        image_path_jpg = os.path.join(self.datadir, image_id + '.jpg')

        if os.path.exists(image_path_png):
            image_path = image_path_png
        elif os.path.exists(image_path_jpg):
            image_path = image_path_jpg
        else:
            raise FileNotFoundError(f"Image file for ID {image_id} not found with .png or .jpg extension.")
        cam_path = os.path.join(self.camdir, image_id+'.npy')
        cam = np.load(cam_path)
        image = Image.open(image_path)
        image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
        image = self.train_augmentation(image)
        image = np.array(image) / 255.
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        label = np.array(int(label))
        name = image_path.split('/')[-1].split('.')[0]
        return image.copy(), label.copy(), cam.copy(), name

# class train_classification_data_4c(data.Dataset):
#     def __init__(self, datadir, maskdir, train_csv, crop_size = (224,224)):
#         self.datadir = datadir
#         self.maskdir = maskdir
#         self.train_csv = train_csv
#         self.files = []
#         self.crop_w, self.crop_h = crop_size
#         self.LUT = {'nv': 0,
#                     'mel': 1,
#                     'bkl': 2, 
#                     'bcc': 3, 
#                     'akiec': 4, 
#                     'vasc': 5, 
#                     'df': 6}
        
#         self.data = pd.read_csv(train_csv)

#         self.data['labels'] = self.data['dx'].map(self.LUT)
#         self.data = self.data[['image_id', 'labels']]

#         self.train_augmentation = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.ToPILImage()
#              ])

            
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         image_id = self.data['image_id'][idx]
#         label = self.data['labels'][idx]
#         image_path = os.path.join(self.datadir, image_id+'.png')
#         mask_path = os.path.join(self.maskdir, image_id+'.png')
#         image = Image.open(image_path)
#         image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         image = self.train_augmentation(image)
#         mask = Image.open(mask_path)
#         mask = mask.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         mask = self.train_augmentation(mask)
#         image = np.array(image) / 255.
#         mask = np.array(mask) 
#         mask= np.where(mask> 127, 255, 0)
#         mask = mask / 255.
#         mask = mask[:, :, np.newaxis]
#         image = np.concatenate((image, mask), axis=-1)
#         image = image.transpose((2, 0, 1))
#         image = image.astype(np.float32)
#         label = np.array(int(label))
#         name = image_path.split('/')[-1].split('.')[0]
#         return image.copy(), label.copy(), name


# class val_classification_data_4c(data.Dataset):
#     def __init__(self, datadir, maskdir, val_csv, crop_size = (224,224)):
#         self.datadir = datadir
#         self.files = []
#         self.maskdir = maskdir
#         self.crop_w, self.crop_h = crop_size
#         self.LUT = {'nv': 0,
#                     'mel': 1,
#                     'bkl': 2, 
#                     'bcc': 3, 
#                     'akiec': 4, 
#                     'vasc': 5, 
#                     'df': 6}
        
#         self.data = pd.read_csv(val_csv)

#         self.data['labels'] = self.data['dx'].map(self.LUT)
#         self.data = self.data[['image_id', 'labels']]

#         self.val_augmentation = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.ToPILImage()
#              ])

            
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         image_id = self.data['image_id'][idx]
#         label = self.data['labels'][idx]
#         image_path = os.path.join(self.datadir, image_id+'.png')
#         mask_path = os.path.join(self.maskdir, image_id+'.png')
#         image = Image.open(image_path)
#         image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         image = self.val_augmentation(image)
#         mask = Image.open(mask_path)
#         mask = mask.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         mask = self.val_augmentation(mask)
#         image = np.array(image) / 255.
#         mask = np.array(mask) 
#         mask= np.where(mask> 127, 255, 0)
#         mask = mask / 255.
#         mask = mask[:, :, np.newaxis]
#         image = np.concatenate((image, mask), axis=-1)
#         image = image.transpose((2, 0, 1))
#         image = image.astype(np.float32)
#         label = np.array(int(label))
#         name = image_path.split('/')[-1].split('.')[0]
#         return image.copy(), label.copy(), name
    

    

# class train_isic_data(data.Dataset):
#     def __init__(self, datadir, train_csv, crop_size = (224,224)):
#         self.datadir = datadir
#         self.train_csv = train_csv
#         self.files = []
#         self.crop_w, self.crop_h = crop_size
#         self.LUT = {'melanoma': 0,
#                     'seborrheic_keratosis': 1,
#                     'none': 2}
        
#         self.data = pd.read_csv(train_csv)

#         self.data['label'] = self.data[['melanoma', 'seborrheic_keratosis']].idxmax(axis=1)
#         self.data.loc[self.data[['melanoma', 'seborrheic_keratosis']].sum(axis=1) == 0, 'label'] = 'none'

#         self.data['labels'] = self.data['label'].map(self.LUT)
#         self.data = self.data[['image_id', 'labels']]

#         self.train_augmentation = transforms.Compose(
#             [transforms.RandomVerticalFlip(p=0.5),
#              transforms.RandomHorizontalFlip(p=0.5),
#              transforms.ToTensor(),
#              transforms.ToPILImage(),
#              transforms.Resize(self.crop_h)
#              ])

            
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         image_id = self.data['image_id'][idx]
#         label = self.data['labels'][idx]
#         image_path = os.path.join(self.datadir, image_id+'.jpg')
#         image = Image.open(image_path)
#         image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         image = self.train_augmentation(image)
#         image = np.array(image) / 255.
#         image = image.transpose((2, 0, 1))
#         image = image.astype(np.float32)
#         label = np.array(int(label))
#         name = image_path.split('/')[-1].split('.')[0]
#         return image.copy(), label.copy(), name
    
# class val_isic_data(data.Dataset):
#     def __init__(self, datadir, train_csv, crop_size = (224,224)):
#         self.datadir = datadir
#         self.train_csv = train_csv
#         self.files = []
#         self.crop_w, self.crop_h = crop_size
#         self.LUT = {'melanoma': 0,
#                     'seborrheic_keratosis': 1,
#                     'none': 2}
        
#         self.data = pd.read_csv(train_csv)

#         self.data['label'] = self.data[['melanoma', 'seborrheic_keratosis']].idxmax(axis=1)
#         self.data.loc[self.data[['melanoma', 'seborrheic_keratosis']].sum(axis=1) == 0, 'label'] = 'none'

#         self.data['labels'] = self.data['label'].map(self.LUT)
#         self.data = self.data[['image_id', 'labels']]

#         self.train_augmentation = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.ToPILImage(),
#              transforms.Resize(self.crop_h)
#              ])

            
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         image_id = self.data['image_id'][idx]
#         label = self.data['labels'][idx]
#         image_path = os.path.join(self.datadir, image_id+'.jpg')
#         image = Image.open(image_path)
#         image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
#         image = self.train_augmentation(image)
#         image = np.array(image) / 255.
#         image = image.transpose((2, 0, 1))
#         image = image.astype(np.float32)
#         label = np.array(int(label))
#         name = image_path.split('/')[-1].split('.')[0]
#         return image.copy(), label.copy(), name