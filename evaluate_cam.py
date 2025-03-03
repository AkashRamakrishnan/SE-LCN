import os
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from datasets import run_eval_cam
from torch.utils.data import DataLoader
from gradcam.utils import visualize_cam, Normalize
from gradcam.gradcam import GradCAM, GradCAMpp
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import warnings
import cv2
import argparse

# Choose either 'symmetry' or 'classification'

parser = argparse.ArgumentParser(description='Evaluate Class Activation Maps (GradCAM++) for a model')
parser.add_argument('--network', type=str, default='classification', help='Use either symmetry or classification')
parser.add_argument('--mask', type=str, default='None', help='If using masked input')
args = parser.parse_args()

if args.network == 'symmetry':
    if args.mask == 'None':
        MODEL_PATH = 'models/SymmetryNet_squeezenet_cv_pretrained/symmetry_net_fold_2_best.pth'
    elif args.mask == 'Segmentation':
        MODEL_PATH = 'models/SymmetryNet_squeezenet_cv_mask/symmetry_net_fold_3_best.pth'
    else:
        MODEL_PATH = 'models/SymmetryNet_squeezenet_cv_lae/symmetry_net_fold_3_best.pth'

elif args.network == 'classification':
    if args.mask == 'None':
        MODEL_PATH = 'models/classification_efficientnet_cv_balanced/symmetry_net_fold_2_best.pth'
    elif args.mask == 'Segmentation':
        MODEL_PATH = 'models/classification_efficientnet_cv_masked/symmetry_net_fold_2_best.pth'
    else:
        MODEL_PATH = 'models/classification_efficientnet_cv_lae/symmetry_net_fold_3_best.pth'


print(f'Network type: {args.network}')
print(f'Using masking strategy {args.mask}')
print(f'Loading model from {MODEL_PATH}')

INPUT_SIZE = '224, 224'
h, w = map(int, INPUT_SIZE.split(','))


if args.network == 'symmetry':
    if args.mask == 'None': 
        img_dir = '../../data/SymDerm_v2/SymDerm_extended'
    if args.mask == 'Segmentation':
        img_dir = '../../data/SymDerm_v2/SymDerm_mask'
    else:
        img_dir = '../../data/SymDerm_v2/SymDerm_lae_mask'
    mask_dir = '../../data/SymDerm_v2/SymDerm_binary_mask'
else:
    if args.mask == 'None':
        img_dir = '../../data/HAM10000_inpaint'
    elif args.mask == 'Segmentation':
        img_dir = '../../data/HAM10000_masks'
    else:
        img_dir = '../../data/HAM10000_lae'
    mask_dir = '../../data/HAM10000_segmentations'
    

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


# Load model
if args.network == 'symmetry':
    squeezenet = torch.load(MODEL_PATH, map_location=device)
    squeezenet.eval()

    squeezenet_model_dict = dict(type='squeezenet', arch=squeezenet, layer_name='features_12_expand3x3_activation', input_size=(h,w))
    squeezenet_gradcampp = GradCAMpp(squeezenet_model_dict, True)
    cam_dict = dict()
    cam_dict['squeezenet'] = [squeezenet_gradcampp]
    loader = DataLoader(run_eval_cam(img_dir, mask_dir, crop_size=(w, h), HAM=False), batch_size=1, shuffle=True)

else:
    efficientnet = torch.load(MODEL_PATH, map_location=device)
    efficientnet.eval()
    efficientnet_model_dict = dict(type='efficientnet', arch=efficientnet, layer_name='features.8.2')
    efficientnet_gradcampp = GradCAMpp(efficientnet_model_dict, True)
    cam_dict = dict()
    cam_dict['efficientnet'] = [efficientnet_gradcampp]
    loader = DataLoader(run_eval_cam(img_dir, mask_dir, crop_size=(w, h), HAM=True), batch_size=1, shuffle=True)

iou_list = []
overlap_list = []
point_count = 0
for index, batch in enumerate(tqdm(loader)):
    data, mask, name = batch
    data = data.to(device)
    mask = mask
    for [gradcam_pp] in cam_dict.values():
        mask_pp0, _ = gradcam_pp(data)
        is_point = pointing_game(mask_pp0.cpu().numpy().squeeze(), mask.numpy().squeeze())
        if is_point:
            point_count += 1
        cam_mask = (mask_pp0 > 0.5).to(torch.uint8)
        mask_dil = increase_segmented_region(mask.numpy().squeeze(), 2)
        mask_arr_dil = Image.fromarray(((mask_dil*255).squeeze()).astype(np.uint8))
        mask_arr = Image.fromarray(((mask*255).squeeze().numpy()).astype(np.uint8))
        cam_mask_arr = Image.fromarray(((cam_mask*255).squeeze().numpy()).astype(np.uint8))


        heatmap_pp0, result_pp0 = visualize_cam(mask_pp0, data)
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

    
avg_iou = np.nanmean(np.array(iou_list))
std_iou = np.nanstd(np.array(iou_list))
avg_overlap = np.mean(np.array(overlap_list))
std_overlap = np.std(np.array(overlap_list))
print(f'Average IOU = {avg_iou:.3f} \u00B1 {std_iou:.3f}')
print(f'Average overlap = {avg_overlap:.3f} \u00B1 {std_overlap:.3f}')
print(f'Point Game Score = {point_count/len(loader):.3f}')
