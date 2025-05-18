import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from datasets import train_classification_cam, val_classification_cam
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score, classification_report, accuracy_score
import argparse
import torch.nn.functional as F
from net.EfficientNetCAM import EfficientNetB4WithELayer, ELayer
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Collect predictions from cross-validated models')
    # parser.add_argument('--model_dir', type=str, required=True, help='Directory containing fold models')
    parser.add_argument('--mask', type=str, required=True, help='Type of masking used: none, mask or lae')
    parser.add_argument('--cam_mask', type=str, required=True, help='Type of CAM masking used: none, mask or lae')
    parser.add_argument('--n_folds', type=int, default=4, help='Number of folds used in cross-validation')
    parser.add_argument('--test_csv', type=str, default='data_files/HAM10000_metadata_balanced.csv', 
                       help='Path to test CSV file')
    args = parser.parse_args()

    if args.mask == 'mask':
        valdir = 'datasets/HAM10000_balanced_masks/'
        # valdir = 'datasets/HAM10000_masks/'

    elif args.mask == 'lae':
        valdir = 'datasets/HAM10000_balanced_lae/'
        # valdir = 'datasets/HAM10000_lae'

    else:
        valdir = 'datasets/HAM10000_balanced/'
        # valdir = 'datasets/HAM10000_inpaint'


    if args.cam_mask == 'mask':
        val_camdir = 'datasets/HAM10000_balanced_masks_cam/'
        # val_camdir = 'datasets/HAM10000_masks_cam'

    elif args.cam_mask == 'lae':
        val_camdir = 'datasets/HAM10000_balanced_lae_cam/'
        # val_camdir = 'datasets/HAM10000_lae_cam'

    else:
        val_camdir = 'datasets/HAM10000_balanced_cam/'
        # val_camdir = 'datasets/HAM10000_inpaint_cam'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = f'model_weights/classification_efficientnet_cv_final_{args.mask}_{args.cam_mask}/'
    
    print(f'Loading model from {model_dir}')
    print(f'Using device: {device}')
    # Load test dataset (adjust paths as needed)
    test_data = pd.read_csv(args.test_csv)
    test_dataset = val_classification_cam(valdir, val_camdir, test_data, crop_size=(256, 256))
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    lookup_table = {'nv':0, 'mel':1, 'bkl':2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
    # Reverse lookup: index -> class name
    index_to_class = {v: k for k, v in lookup_table.items()}

    # Initialize dataframe to store results
    results = pd.DataFrame()
    
    for fold in range(1, args.n_folds+1):
        print(f'Processing fold {fold}...')
        
        # Load model
        model_path = os.path.join(model_dir, f'symmetry_net_fold_{fold}_best.pth')
        model = torch.load(model_path, map_location=device)
        model.eval()
        
        # Collect predictions
        fold_preds = []
        fold_pred_names = []
        image_names = []
        true_labels = []
        
        with torch.no_grad():
            # for inputs, labels, cams, names in tqdm(test_loader):
            for inputs, labels, cams, names in test_loader:
                inputs = inputs.to(device)
                cams = cams.unsqueeze(1).to(device)
                
                outputs = model(inputs, cams)
                _, preds = torch.max(outputs, 1)
                
                fold_preds.extend(preds.cpu().numpy())
                fold_pred_names.extend([index_to_class.get(p, 'Unknown') for p in preds.cpu().numpy()])
                true_labels.extend(labels.numpy())
                image_names.extend(names)

        # Add image names if not already present
        if 'image_name' not in results.columns:
            results['image_name'] = image_names

        # Store results
        results['true_label'] = true_labels  # Will overwrite each time, but values are same
        results[f'fold_{fold}_pred'] = fold_pred_names
        
        

    # Save results
    output_csv = os.path.join('predictions',f'predictions_balanced_train_{args.mask}_{args.cam_mask}.csv')
    results.to_csv(output_csv, index=False)
    print(f'Saved predictions to {output_csv}')

if __name__ == '__main__':
    main()
