import os
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from datasets import run_eval_cam_final, run_eval_cam
from torch.utils.data import DataLoader
from gradcam.utils import visualize_cam, Normalize
from gradcam.gradcam import GradCAM, GradCAMppfinal, GradCAMpp
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import warnings
import cv2
import argparse
from net.EfficientNetCAM import EfficientNetB4WithELayer, ELayer



def generate_cam(mask, sym_model, device):
    if mask == 'None':
        img_dir = 'cam_dir/HAM10000_inpaint'
        if sym_model == 'None':
            MODEL_PATH = 'models/classification_efficientnet_cv_final_none_none/symmetry_net_fold_4_best.pth'
            camdir = '../../data/HAM10000_inpaint_cam/'
            
        elif sym_model == 'Segmentation':
            MODEL_PATH = 'models/classification_efficientnet_cv_final_none_mask/symmetry_net_fold_2_best.pth'
            camdir = '../../data/HAM10000_masks_cam/'

        elif sym_model == 'LAE':
            MODEL_PATH = 'models/classification_efficientnet_cv_final_none_lae/symmetry_net_fold_3_best.pth'
            camdir = '../../data/HAM10000_lae_cam/'

        else:
            MODEL_PATH = 'models/classification_efficientnet_cv_balanced/symmetry_net_fold_2_best.pth'
            camdir = None



    elif mask == 'Segmentation':
        img_dir = 'cam_dir/HAM10000_masks'
        if sym_model == 'None':
            MODEL_PATH = 'models/classification_efficientnet_cv_final_mask_none/symmetry_net_fold_1_best.pth'
            camdir = '../../data/HAM10000_inpaint_cam/'
            
        elif sym_model == 'Segmentation':
            MODEL_PATH = 'models/classification_efficientnet_cv_final_mask_mask/symmetry_net_fold_2_best.pth'
            camdir = '../../data/HAM10000_masks_cam/'

        elif sym_model == 'LAE':
            MODEL_PATH = 'models/classification_efficientnet_cv_final_mask_lae/symmetry_net_fold_4_best.pth'
            camdir = '../../data/HAM10000_lae_cam/'

        else:
            MODEL_PATH = 'models/classification_efficientnet_cv_masked/symmetry_net_fold_2_best.pth'
            camdir = None



            
    else:
        img_dir = 'cam_dir/HAM10000_lae'
        if sym_model == 'None':
            MODEL_PATH = 'models/classification_efficientnet_cv_final_lae_none/symmetry_net_fold_2_best.pth'
            camdir = '../../data/HAM10000_inpaint_cam/'

        elif sym_model == 'Segmentation':
            MODEL_PATH = 'models/classification_efficientnet_cv_final_lae_mask/symmetry_net_fold_2_best.pth'
            camdir = '../../data/HAM10000_masks_cam/'

        elif sym_model == 'LAE':
            MODEL_PATH = 'models/classification_efficientnet_cv_final_lae_lae/symmetry_net_fold_2_best.pth'
            camdir = '../../data/HAM10000_lae_cam/'

        else:
            MODEL_PATH = 'models/classification_efficientnet_cv_lae/symmetry_net_fold_3_best.pth'
            camdir = None


    print('*-'*20)
    print(f'Using masking strategy {mask}')
    print(f'Using symmetry masking {sym_model}')
    print(f'Loading model from {MODEL_PATH}')
    output_path = os.path.join('model_cams/', f'{mask}_{sym_model}.png')

    INPUT_SIZE = '224, 224'
    h, w = map(int, INPUT_SIZE.split(','))

    mask_dir = '../../data/HAM10000_segmentations'

    torch.manual_seed(42)
    np.random.seed(42)

    efficientnet = torch.load(MODEL_PATH, map_location=device)
    efficientnet.eval()
    if camdir != None:
        efficientnet_model_dict = dict(type='efficientnet', arch=efficientnet, layer_name='e_layer.relu')
        efficientnet_gradcampp = GradCAMppfinal(efficientnet_model_dict, True)
        loader = DataLoader(run_eval_cam_final(img_dir, mask_dir, camdir, crop_size=(w, h)), batch_size=1, shuffle=False)

    else:
        efficientnet_model_dict = dict(type='efficientnet', arch=efficientnet, layer_name='features.8.2')
        efficientnet_gradcampp = GradCAMpp(efficientnet_model_dict, True)
        loader = DataLoader(run_eval_cam(img_dir, mask_dir, crop_size=(w, h), HAM=True), batch_size=1, shuffle=True)


    cam_dict = dict()
    cam_dict['efficientnet'] = [efficientnet_gradcampp]

    if camdir != None:
        for index, batch in enumerate(loader):
            data, mask, cams, name = batch
            data = data.to(device)
            cams = cams.to(device)
            cams = cams.unsqueeze(1)
            mask = mask
            print(name)
            images = []
            for [gradcam_pp] in cam_dict.values():
                mask_pp0, _ = efficientnet_gradcampp(data, cams)
                heatmap_pp0, result_pp0 = visualize_cam(mask_pp0, data)
                is_point = pointing_game(mask_pp0.cpu().numpy().squeeze(), mask.numpy().squeeze())
                if is_point:
                    print('Point lies in the region')
                cam_mask = (mask_pp0 > 0.5).to(torch.uint8)
                mask_dil = increase_segmented_region(mask.numpy().squeeze(), 2)
                mask_dil = mask_dil.flatten()
                cam_mask = cam_mask.flatten().cpu()
                iou = jaccard_score(mask_dil, cam_mask)
                print('IoU =', iou)
                overlap = cam_mask * mask_dil
                total_mask_points = mask_dil.sum().item()
                total_overlap_points = overlap.sum().item()
                total_cam_points = cam_mask.sum().item()
                if total_mask_points > 0:
                    percentage_overlap = total_overlap_points/total_cam_points
                else:
                    percentage_overlap = 0
                print('Percentage overlap = ', percentage_overlap)
                images.append(torch.stack([data.squeeze().cpu(), heatmap_pp0], 0))
            break
        images = make_grid(torch.cat(images, 0), nrow=5)
        save_image(images, output_path)

    else:
        for index, batch in enumerate(loader):
            data, mask, name = batch
            data = data.to(device)
            mask = mask
            print(name)
            images = []
            for [gradcam_pp] in cam_dict.values():
                mask_pp0, _ = efficientnet_gradcampp(data)
                heatmap_pp0, result_pp0 = visualize_cam(mask_pp0, data)
                is_point = pointing_game(mask_pp0.cpu().numpy().squeeze(), mask.numpy().squeeze())
                if is_point:
                    print('Point lies in the region')
                cam_mask = (mask_pp0 > 0.5).to(torch.uint8)
                mask_dil = increase_segmented_region(mask.numpy().squeeze(), 2)
                mask_dil = mask_dil.flatten()
                cam_mask = cam_mask.flatten().cpu()
                iou = jaccard_score(mask_dil, cam_mask)
                print('IoU =', iou)
                overlap = cam_mask * mask_dil
                total_mask_points = mask_dil.sum().item()
                total_overlap_points = overlap.sum().item()
                total_cam_points = cam_mask.sum().item()
                if total_mask_points > 0:
                    percentage_overlap = total_overlap_points/total_cam_points
                else:
                    percentage_overlap = 0
                print('Percentage overlap = ', percentage_overlap)
                images.append(torch.stack([data.squeeze().cpu(), heatmap_pp0], 0))
            break
        images = make_grid(torch.cat(images, 0), nrow=5)
        save_image(images, output_path)






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


if __name__ == "__main__":
    # Suppress specific warning by category
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    generate_cam('None', 'None', device)
    generate_cam('None', 'Segmentation', device)
    generate_cam('None', 'LAE', device)
    generate_cam('None', 'NoSym', device)
    generate_cam('Segmentation', 'None', device)
    generate_cam('Segmentation', 'Segmentation', device)
    generate_cam('Segmentation', 'LAE', device)
    generate_cam('Segmentation', 'NoSym', device)
    generate_cam('LAE', 'None', device)
    generate_cam('LAE', 'Segmentation', device)
    generate_cam('LAE', 'LAE', device)
    generate_cam('LAE', 'NoSym', device)