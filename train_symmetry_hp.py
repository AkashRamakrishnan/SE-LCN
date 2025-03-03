import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
import os
import optuna
from torch.utils.data import DataLoader
from datasets import train_symmetry_data, val_symmetry_data
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup argument parser
parser = argparse.ArgumentParser(description='Train a neural network model on image data using Bayesian optimization with early stopping.')
parser.add_argument('--mask', type=str, default='None', help='Type of input masking (Either None, Segmentation, or LAE)')
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

    # Data loading based on mask argument
    if args.mask == 'None':
        datadir_train = '../../data/SymDerm_v2/SymDerm_extended/'
        datadir_val = '../../data/SymDerm_v2/SymDerm_extended/'

    if args.mask == 'Segmentation':
        datadir_train = '../../data/SymDerm_v2/SymDerm_mask/'
        datadir_val = '../../data/SymDerm_v2/SymDerm_mask/'

    else:
        datadir_train = '../../data/SymDerm_v2/SymDerm_lae_mask' 
        datadir_val = '../../data/SymDerm_v2/SymDerm_lae_mask'

    train_csv = 'data_files/SymDerm_v2_train.csv'

    train_data = pd.read_csv(train_csv)
    train_data, val_data = train_test_split(train_data, test_size=0.25, stratify=train_data['labels_symmetry'], random_state=42)
    train_data, val_data = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    train_dataset = train_symmetry_data(datadir_train, train_data, crop_size=(256, 256))
    val_dataset = val_symmetry_data(datadir_val, val_data, crop_size=(256, 256))

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model definition
    model = models.squeezenet1_0(pretrained=False)
    model.load_state_dict(torch.load('models/squeezenet1_0-b66bff10.pth', map_location=device))
    model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
    model.num_classes = 2
    model = model.to(device)

    # Loss and optimizer
    if use_weighted_loss:
        label_counts = train_data['labels_symmetry'].value_counts()
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
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels, _ in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        train_loss = running_loss / len(trainloader)
        train_accuracy = correct_predictions / total_predictions
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, _ in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
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
study.optimize(objective, n_trials=30)  # You can set the number of trials

print("Best trial:")
trial = study.best_trial
print(f" Value: {trial.value}")
print(" Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
