import os
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from datasets import run_segmentation_data
from torch.utils.data import DataLoader
from gradcam.utils import visualize_cam, Normalize
from gradcam.gradcam import GradCAM, GradCAMpp
from tqdm import tqdm
import warnings
import cv2
import argparse

parser = argparse.ArgumentParser(description='Generate CAMs for a dataset using symmetry classification network')
parser.add_argument('--mask', type=str, default='None', help='If using masked input')
args = parser.parse_args()

if args.mask == 'None':
    MODEL_PATH = 'models/SymmetryNet_squeezenet_cv_pretrained/symmetry_net_fold_2_best.pth'
elif args.mask == 'Segmentation':
    MODEL_PATH = 'models/SymmetryNet_squeezenet_cv_mask/symmetry_net_fold_3_best.pth'
else:
    MODEL_PATH = 'models/SymmetryNet_squeezenet_cv_lae/symmetry_net_fold_3_best.pth'

print(f'Using masking strategy {args.mask}')
print(f'Loading model from {MODEL_PATH}')

INPUT_SIZE = '224, 224'
h, w = map(int, INPUT_SIZE.split(','))

if args.mask == 'None':
    img_dir = '../../data/SymDerm_v2/SymDerm_extended'
elif args.mask == 'Segmentation':
    img_dir = '../../data/SymDerm_v2/SymDerm_mask'
else:
    img_dir = '../../data/SymDerm_v2/SymDerm_lae_mask'
mask_dir = '../../data/SymDerm_v2/SymDerm_binary_mask'


save_dir = img_dir+'_cam/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Suppress specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(42)
np.random.seed(42)


# Load model
squeezenet = torch.load(MODEL_PATH, map_location=device)
squeezenet.eval()

squeezenet_model_dict = dict(type='squeezenet', arch=squeezenet, layer_name='features_12_expand3x3_activation', input_size=(h,w))
# squeezenet_gradcam = GradCAM(squeezenet_model_dict, True)
squeezenet_gradcampp = GradCAMpp(squeezenet_model_dict, True)

cam_dict = dict()
cam_dict['squeezenet'] = [squeezenet_gradcampp]

loader = DataLoader(run_segmentation_data(img_dir, crop_size=(w, h)), batch_size=1, shuffle=True)

iou_list = []
overlap_list = []
point_count = 0
for index, batch in enumerate(tqdm(loader)):
    data, name = batch
    data = data.to(device)
    for [gradcam_pp] in cam_dict.values():
        mask_pp0, _ = gradcam_pp(data)
        cam = mask_pp0.cpu().numpy().squeeze()
        path = os.path.join(save_dir, name[0])
        np.save(path, cam)
        
