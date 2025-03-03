import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
import os
import optuna
from torch.utils.data import DataLoader
from datasets import train_classification_cam, val_classification_cam
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from net.EfficientNetCAM import EfficientNetB4WithELayer, ELayer
import time

# Setup argument parser
parser = argparse.ArgumentParser(description='Train a neural network model on image data using Bayesian optimization with early stopping.')
parser.add_argument('--mask', type=str, default='None', help='If using masked input')
parser.add_argument('--cam_mask', type=str, default='None', help='Masking that was used in symmetry classification models that is generated CAMs')
args = parser.parse_args()
# Define the objective function for hyperparameter tuning
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    use_weighted_loss = trial.suggest_categorical('use_weighted_loss', [True, False])

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES_CLS = 7
    if args.mask == 'mask':
        datadir = '../../data/HAM10000_balanced_masks/'
        valdir = '../../data/HAM10000_masks/'

    elif args.mask == 'lae':
        datadir = '../../data/HAM10000_balanced_lae/'
        valdir = '../../data/HAM10000_lae'

    else:
        datadir = '../../data/HAM10000_balanced/'
        valdir = '../../data/HAM10000_inpaint'


    if args.cam_mask == 'mask':
        camdir = '../../data/HAM10000_balanced_masks_cam/'
        val_camdir = '../../data/HAM10000_masks_cam'

    elif args.cam_mask == 'lae':
        camdir = '../../data/HAM10000_balanced_lae_cam/'
        val_camdir = '../../data/HAM10000_lae_cam'

    else:
        camdir = '../../data/HAM10000_balanced_cam/'
        val_camdir = '../../data/HAM10000_inpaint_cam'


    train_csv = 'data_files/HAM10000_metadata_balanced.csv'
    test_csv = 'data_files/HAM10000_metadata_test.csv'

    train_data = pd.read_csv(train_csv)
    train_data, val_data = train_test_split(train_data, test_size=0.25, stratify=train_data['dx'], random_state=42)
    train_data, val_data = train_data.reset_index(drop=True), val_data.reset_index(drop=True)

    train_dataset = train_classification_cam(datadir, camdir, train_data, crop_size=(256, 256))
    val_dataset = val_classification_cam(datadir, camdir, val_data, crop_size=(256, 256))

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = EfficientNetB4WithELayer(num_classes=NUM_CLASSES_CLS, cam_channels=1, device=device).to(device)

    #Loss and optimizer
    if use_weighted_loss:
        label_counts = train_data['dx'].value_counts()
        weights = 1.0 / label_counts
        weights_tensor = torch.tensor(weights.sort_index().values, dtype=torch.float)
        loss_function = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    else:
        loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_accuracy = 0
    epochs_no_improve = 0
    patience = 10  # Early stopping patience

    # Training and validation loop with early stopping
    for epoch in range(50):  # Maximum epochs
        t = time.time()
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels, cams, _ in trainloader:
            inputs, labels, cams = inputs.to(device), labels.to(device), cams.to(device)
            cams = cams.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs, cams)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        train_loss = running_loss / len(trainloader)
        train_accuracy = correct_predictions / total_predictions
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Epoch Duraction: {time.time() - t:.2} seconds')

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, cams, _ in valloader:
                inputs, labels, cams = inputs.to(device), labels.to(device), cams.to(device)
                cams = cams.unsqueeze(1)
                outputs = model(inputs, cams)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_accuracy = correct / total
        print(f'Epoch {epoch+1}, Val Accuracy: {val_accuracy:.4f}')
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return best_val_accuracy

# Running the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)  # You can set the number of trials

print("Best trial:")
trial = study.best_trial
print(f" Value: {trial.value}")
print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
