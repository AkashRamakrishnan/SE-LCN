import torch
import numpy as np
import os
from net.models import deeplabv3plus
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datasets import test_segmentation_data, val_segmentation_data, run_segmentation_data
from torch.utils import data
from unet import UNet
from PIL import Image
from tqdm import tqdm
import csv
import cv2

MODEL_PATH = 'model_weights/Segmentation_Deeplab_1/CoarseSN_e61.pth'  
INPUT_SIZE = '256, 256'
NUM_CLASSES = 2
BATCH_SIZE = 1

w, h = map(int, INPUT_SIZE.split(','))

def generate_masked_img(testloader, model, outdir, device):
    if not os.path.isdir(outdir):
            os.mkdir(outdir)
    for index, batch in enumerate(tqdm(testloader)):
        data, name = batch
        data = data.to(device)
        # print(data[0].shape)
        model.to(device)

        model.eval()
        with torch.no_grad():
            pred = model(data)
        
        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)
        pred_arg = pred_arg.astype(np.uint8)
        data = data[0].cpu().data.numpy().transpose(1, 2, 0)
        mask_img = np.zeros_like(data)
        for channel in range(3):
            mask_img[:, :, channel] = data[:, :, channel] * pred_arg
        # # mask_array = pred_arg * 255
        # # mask = Image.fromarray(mask_array)
        # mask_img = mask_img.reshape(224, 224, 3)
        # print(mask_img.shape)
        mask_img = Image.fromarray((mask_img*255).astype(np.uint8))
        mask_img = mask_img.resize((640, 450), Image.BICUBIC)
        mask_img.save(os.path.join(outdir, name[0] + '.jpg'))

def generate_masks(testloader, model, outdir):
    for index, batch in enumerate(tqdm(testloader)):
        data, name = batch
        data = data.cuda()
        # print(data[0].shape)

        model.eval()
        with torch.no_grad():
            pred = model(data)
        
        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)
        pred_arg = pred_arg.astype(np.uint8)
        mask_array = pred_arg * 255
        mask_arr = Image.fromarray(mask_array)
        mask_arr = mask_arr.resize((640, 450), Image.BICUBIC)
        mask_arr.save(os.path.join(outdir, name[0] + '.png'))
        
def generate_lae(testloader, model, outdir, device):
    if not os.path.isdir(outdir):
            os.mkdir(outdir)
    for index, batch in enumerate(tqdm(testloader)):
        data, name = batch
        data = data.to(device)
        # print(data[0].shape)

        model.eval()
        with torch.no_grad():
            pred = model(data)
        
        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)
        pred_arg = pred_arg.astype(np.uint8)
        mask_arr = pred_arg * 255
        contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        data = data[0].cpu().data.numpy().transpose(1, 2, 0)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            new_x = max(0, x - int(0.2*w))
            new_y = max(0, y - int(0.2*h))
            new_w = int(1.4*w)
            new_h = int(1.4*h)
            new_w = min(new_w, mask_arr.shape[1] - new_x)
            new_h = min(new_h, mask_arr.shape[0] - new_y)
            mask = np.zeros_like(data)
            mask[new_y:new_y+new_h, new_x:new_x+new_w] = [255, 255, 255]
            masked_image = np.where(mask == [255, 255, 255], data, [0, 0, 0])
            mask_arr = Image.fromarray((masked_image*255).astype(np.uint8))
            mask_arr = mask_arr.resize((640, 450), Image.BICUBIC)
            mask_arr.save(os.path.join(outdir, name[0] + '.jpg'))
        else:
            img = Image.fromarray((data*255).astype(np.uint8))
            img = img.resize((640, 450), Image.BICUBIC)
            img.save(os.path.join(outdir, name[0] + '.jpg'))

def generate_mask_symderm(testloader, model, outdir):
    for index, batch in enumerate(tqdm(testloader)):
        data, name = batch
        data = data.cuda()
        # print(data[0].shape)

        model.eval()
        with torch.no_grad():
            pred = model(data)
        
        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)
        pred_arg = pred_arg.astype(np.uint8)
        data = data[0].cpu().data.numpy().transpose(1, 2, 0)
        mask_img = np.zeros_like(data)
        for channel in range(3):
            mask_img[:, :, channel] = data[:, :, channel] * pred_arg
        # # mask_array = pred_arg * 255
        # # mask = Image.fromarray(mask_array)
        # mask_img = mask_img.reshape(224, 224, 3)
        # print(mask_img.shape)
        mask_img = Image.fromarray((mask_img*255).astype(np.uint8))
        mask_img = mask_img.resize((640, 450), Image.BICUBIC)
        mask_img.save(os.path.join(outdir, name[0] + '.png'))

def generate_binary_masks(testloader, model, outdir):
    for index, batch in enumerate(tqdm(testloader)):
        data, name = batch
        data = data.cuda()
        # print(data[0].shape)

        model.eval()
        with torch.no_grad():
            pred = model(data)
        
        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)
        pred_arg = pred_arg.astype(np.uint8)
        mask_array = pred_arg * 255
        mask_arr = Image.fromarray(mask_array)
        mask_arr = mask_arr.resize((640, 450), Image.BICUBIC)
        mask_arr.save(os.path.join(outdir, name[0] + '.png'))
    
def generate_lae_masks(testloader, model, outdir, device):
    for index, batch in enumerate(tqdm(testloader)):
        data, name = batch
        data = data.to(device)
        # print(data[0].shape)

        model.eval()
        with torch.no_grad():
            pred = model(data)
        
        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)
        pred_arg = pred_arg.astype(np.uint8)
        mask_arr = pred_arg * 255
        contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        data = data[0].cpu().data.numpy().transpose(1, 2, 0)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            new_x = max(0, x - int(0.2*w))
            new_y = max(0, y - int(0.2*h))
            new_w = int(1.4*w)
            new_h = int(1.4*h)
            new_w = min(new_w, mask_arr.shape[1] - new_x)
            new_h = min(new_h, mask_arr.shape[0] - new_y)
            mask = np.zeros_like(mask_arr)
            mask[new_y:new_y+new_h, new_x:new_x+new_w] = 255
            mask_arr = Image.fromarray((mask).astype(np.uint8))
            mask_arr = mask_arr.resize((640, 450), Image.BICUBIC)
            mask_arr.save(os.path.join(outdir, name[0] + '.jpg'))
        else:
            mask = np.zeros_like(mask_arr)
            mask[:, :] = 255
            mask_arr = Image.fromarray((mask).astype(np.uint8))
            mask_arr = mask_arr.resize((640, 450), Image.BICUBIC)
            mask_arr.save(os.path.join(outdir, name[0] + '.jpg'))


def generate_lae_coordinates(testloader, model, output_csv, device):
    # Open a CSV file to write the bounding box coordinates
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'new_x', 'new_y', 'new_w', 'new_h'])

        for index, batch in enumerate(tqdm(testloader)):
            data, name = batch
            data = data.to(device)

            model.eval()
            with torch.no_grad():
                pred = model(data)
            
            pred = torch.softmax(pred, dim=1).cpu().data.numpy()
            pred_arg = np.argmax(pred[0], axis=0)
            pred_arg = pred_arg.astype(np.uint8)
            mask_arr = pred_arg * 255
            contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            original_shape = mask_arr.shape
            resize_scale_x = 640 / original_shape[1]
            resize_scale_y = 450 / original_shape[0]

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                new_x = max(0, x - int(0.2 * w))
                new_y = max(0, y - int(0.2 * h))
                new_w = int(1.4 * w)
                new_h = int(1.4 * h)
                new_w = min(new_w, original_shape[1] - new_x)
                new_h = min(new_h, original_shape[0] - new_y)

                # Adjust coordinates for the resized image
                new_x = int(new_x * resize_scale_x)
                new_y = int(new_y * resize_scale_y)
                new_w = int(new_w * resize_scale_x)
                new_h = int(new_h * resize_scale_y)
            else:
                new_x, new_y, new_w, new_h = 0, 0, 640, 450  # Default to full image if no contour is found
            
            # Write the adjusted bounding box to the CSV file
            writer.writerow([name[0], new_x, new_y, new_w, new_h])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = torch.load(MODEL_PATH, map_location=device)
    print("Model loaded successfully.")
    model = model.to(device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    data_test_root = '/home/s3075451/'
    data_test_list = 'data_files/HAM10000_seg.txt'
    # testloader = data.DataLoader(val_segmentation_data(data_test_root, data_test_list, crop_size=(w, h)), batch_size=1, shuffle=False,
                                #  num_workers=8, pin_memory=True)
    mask_path = 'SegOutput/masks'
    masked_imgs_path = 'SegOutput/masked_imgs'
    # lae_path = 'SegOutput/lae'
    lae_path = '../../data/HAM10000_lae_masks'
    symderm_path = '../../data/SymDerm_v2/SymDerm_extended'
    root_path = '../../data/ISIC_2019'
    symderm_outpath = '../../data/SymDerm_v2/SymDerm_lae_mask'
    # outpath = '../../data/ISIC_2019_lae'

    input_path = 'datasets/HAM10000_balanced/'
    outpath = 'datasets/HAM10000_balanced_lae'

    symderm_loader = data.DataLoader(run_segmentation_data(input_path, crop_size=(w, h)), batch_size=1, shuffle=False,
                                 num_workers=8, pin_memory=True)

    # generate_masks(testloader, model, mask_path)
    # generate_masked_img(symderm_loader, model, outpath, device)
    generate_lae(symderm_loader, model, outpath, device)
    # generate_binary_masks(symderm_loader, model, symderm_outpath)
    # generate_lae_masks(symderm_loader, model, symderm_outpath)
    # generate_lae_coordinates(symderm_loader, model, 'coordinates.csv')



if __name__ == '__main__':
    main()
