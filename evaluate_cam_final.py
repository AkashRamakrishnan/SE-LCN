import os
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from datasets import run_eval_cam_final
from torch.utils.data import DataLoader
from gradcam.utils import visualize_cam, Normalize
from gradcam.gradcam import GradCAM, GradCAMppfinal
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import warnings
import cv2
import argparse
from net.EfficientNetCAM import EfficientNetB4WithELayer, ELayer

parser = argparse.ArgumentParser(description='Evaluate Class Activation Maps (GradCAM++) for a model')
parser.add_argument('--mask', type=str, default='None', help='If using masked input. Select None/Segmentation/LAE')
parser.add_argument('--sym_model', type=str, default='None', help='Masking strategy used in symmetry model used to generated CAMs. None/Segmentation/LAE')
args = parser.parse_args()


if args.mask == 'None':
    img_dir = '../../data/HAM10000_inpaint'
    if args.sym_model == 'None':
        MODEL_PATH = 'models/classification_efficientnet_cv_final_none_none/symmetry_net_fold_4_best.pth'
        camdir = '../../data/HAM10000_inpaint_cam/'
        
    elif args.sym_model == 'Segmentation':
        MODEL_PATH = 'models/classification_efficientnet_cv_final_none_mask/symmetry_net_fold_2_best.pth'
        camdir = '../../data/HAM10000_masks_cam/'

    else:
        MODEL_PATH = 'models/classification_efficientnet_cv_final_none_lae/symmetry_net_fold_3_best.pth'
        camdir = '../../data/HAM10000_lae_cam/'


elif args.mask == 'Segmentation':
    img_dir = '../../data/HAM10000_masks'
    if args.sym_model == 'None':
        MODEL_PATH = 'models/classification_efficientnet_cv_final_mask_none/symmetry_net_fold_1_best.pth'
        camdir = '../../data/HAM10000_inpaint_cam/'
        
    elif args.sym_model == 'Segmentation':
        MODEL_PATH = 'models/classification_efficientnet_cv_final_mask_mask/symmetry_net_fold_2_best.pth'
        camdir = '../../data/HAM10000_masks_cam/'

    else:
        MODEL_PATH = 'models/classification_efficientnet_cv_final_mask_lae/symmetry_net_fold_4_best.pth'
        camdir = '../../data/HAM10000_lae_cam/'

        
else:      #For LAE masking
    img_dir = '../../data/HAM10000_lae'
    if args.sym_model == 'None':
        MODEL_PATH = 'models/classification_efficientnet_cv_final_lae_none/symmetry_net_fold_2_best.pth'
        camdir = '../../data/HAM10000_inpaint_cam/'

    elif args.sym_model == 'Segmentation':
        MODEL_PATH = 'models/classification_efficientnet_cv_final_lae_mask/symmetry_net_fold_2_best.pth'
        camdir = '../../data/HAM10000_masks_cam/'

    else:
        MODEL_PATH = 'models/classification_efficientnet_cv_final_lae_lae/symmetry_net_fold_2_best.pth'
        camdir = '../../data/HAM10000_lae_cam/'



print(f'Using masking strategy {args.mask}')
print(f'Loading model from {MODEL_PATH}')
print(img_dir)
print(camdir)
INPUT_SIZE = '224, 224'
h, w = map(int, INPUT_SIZE.split(','))
# img_dir = 'gradcam/images/'

mask_dir = '../../data/ISBI2016_ISIC_Part1_Training_GroundTruth'

def increase_segmented_region(mask, percentage):
    # Calculate the number of iterations based on the desired increase percentage
    # The exact number of iterations required can be tuned
    iterations = int(np.ceil(percentage))
    # print(mask.shape)
    # Define the structuring element (kernel) for dilation
    kernel_size = int(np.ceil(mask.shape[1] * 0.05))  # Adjust kernel size based on your needs
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)

    return dilated_mask


def pointing_game(cam, segmentation_mask):

    max_pos = np.unravel_index(np.argmax(cam), cam.shape)
    y, x = max_pos

    is_point_inside = segmentation_mask[y, x] > 0

    return is_point_inside


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Suppress specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(42)
np.random.seed(42)

# Load Model
efficientnet = torch.load(MODEL_PATH, map_location=device)
efficientnet.eval()
efficientnet_model_dict = dict(type='efficientnet', arch=efficientnet, layer_name='e_layer.relu')
efficientnet_gradcampp = GradCAMppfinal(efficientnet_model_dict, True)

cam_dict = dict()
cam_dict['efficientnet'] = [efficientnet_gradcampp]

loader = DataLoader(run_eval_cam_final(img_dir, mask_dir, camdir, crop_size=(w, h)), batch_size=1, shuffle=True)

iou_list = []
overlap_list = []
point_count = 0
for index, batch in enumerate(loader):
    if not batch: 
        continue
    if index % 1000 == 0:
        print(f'{index}/{len(loader)}')
    data, mask, cams, name = batch
    data = data.to(device)
    cams = cams.to(device)
    cams = cams.unsqueeze(1)
    mask = mask
    for [gradcam_pp] in cam_dict.values():
        mask_pp0, _ = efficientnet_gradcampp(data, cams)
        is_point = pointing_game(mask_pp0.cpu().numpy().squeeze(), mask.numpy().squeeze())
        if is_point:
            point_count += 1
        cam_mask = (mask_pp0 > 0.5).to(torch.uint8)
        mask_dil = increase_segmented_region(mask.numpy().squeeze(), 2)
        mask_dil = mask_dil.flatten()
        cam_mask = cam_mask.flatten().cpu()
        iou = jaccard_score(mask_dil, cam_mask)
        iou_list.append(iou)
        overlap = cam_mask * mask_dil
        total_mask_points = mask_dil.sum().item()
        total_overlap_points = overlap.sum().item()
        total_cam_points = cam_mask.sum().item()
        if total_mask_points > 0:
            percentage_overlap = total_overlap_points/total_cam_points
        else:
            percentage_overlap = 0
        overlap_list.append(percentage_overlap)


avg_iou = np.mean(np.array(iou_list))
std_iou = np.std(np.array(iou_list))
avg_overlap = np.mean(np.array(overlap_list))
std_overlap = np.std(np.array(overlap_list))
print(f'Average IOU = {avg_iou:.3f} \u00B1 {std_iou:.3f}')
print(f'Average overlap = {avg_overlap:.3f} \u00B1 {std_overlap:.3f}')
print(f'Point Game Score = {point_count/len(loader):.3f}')
