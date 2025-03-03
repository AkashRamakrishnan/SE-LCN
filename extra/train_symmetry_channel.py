import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from net.models import DynamicCNNModel, SymNet, Xception_dilation
import matplotlib.pyplot as plt
import os
from datasets import train_symmetry_data_4c, val_symmetry_data_4c
from torch.utils import data
from torchvision import models

import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description='Train a neural network model on image data.')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model to save checkpoints and plots.')
args = parser.parse_args()

# Check if CUDA (GPU support) is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
learning_rate = 0.0001
num_epochs = 100
patience = 10  # For early stopping
INPUT_SIZE = '256, 256'
h, w = map(int, INPUT_SIZE.split(','))
NUM_CLASSES_CLS = 2
BATCH_SIZE = 8
MODEL_NAME = args.model_name

model = models.squeezenet1_0(pretrained=False)

# Modify the first convolution layer to accept 4-channel input
# SqueezeNet's first conv layer is named 'features[0]' and has 'in_channels' as an argument
first_conv_layer = model.features[0]
model.features[0] = nn.Conv2d(4, first_conv_layer.out_channels, 
                              kernel_size=first_conv_layer.kernel_size, 
                              stride=first_conv_layer.stride, 
                              padding=first_conv_layer.padding)

# Modify the final convolution layer to have the correct number of classes
# SqueezeNet's final conv layer is part of the classifier attribute
final_conv_layer = model.classifier[1]
model.classifier[1] = nn.Conv2d(final_conv_layer.in_channels, NUM_CLASSES_CLS,
                                kernel_size=final_conv_layer.kernel_size,
                                stride=final_conv_layer.stride, 
                                padding=final_conv_layer.padding)

# Update the number of classes
model.num_classes = NUM_CLASSES_CLS

model = model.to(device)

loss_function = nn.CrossEntropyLoss()
val_loss_function = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Early stopping setup
best_loss = float('inf')
epochs_no_improve = 0

# Create a directory for saving plots if it doesn't exist
plots_dir = 'training_plots'
os.makedirs(plots_dir, exist_ok=True)

# For plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Load Datasets
datadir = '../../data/SymDerm_v2/SymDerm_extended/'
maskdir = '../../data/SymDerm_v2/SymDerm_binary_mask'
train_csv = 'data_files/SymDerm_v2_train.csv'
trainloader = data.DataLoader(train_symmetry_data_4c(datadir, maskdir, train_csv, crop_size=(h, w)),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)


val_csv = 'data_files/SymDerm_v2_val.csv'
valloader = data.DataLoader(val_symmetry_data_4c(datadir, maskdir, val_csv, crop_size=(h, w)), batch_size=1, shuffle=False,
                            num_workers=8,
                            pin_memory=True)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels, names in trainloader:
        # Move inputs and labels to the correct device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    train_loss = running_loss / len(trainloader)
    train_accuracy = correct_predictions / total_predictions
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0

    with torch.no_grad():
        for inputs, labels, names in valloader:
            # Move inputs and labels to the correct device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = val_loss_function(outputs, labels)

            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_correct_predictions += (predicted == labels).sum().item()
            val_total_predictions += labels.size(0)

    val_loss = val_running_loss / len(valloader)
    val_accuracy = val_correct_predictions / val_total_predictions
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    path = 'models/'+ MODEL_NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    # Early stopping and saving the best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model, os.path.join(path, f'symmetry_net_ep{epoch+1}.pth'))
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print('Early stopping triggered.')
            break

    # Plotting and saving plots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path, f'plot.png'))
    plt.close()
