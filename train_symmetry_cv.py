import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from net.models import DynamicCNNModel, SymNet, Xception_dilation
import matplotlib.pyplot as plt
import os
from datasets import train_symmetry_data, val_symmetry_data
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score, classification_report
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description='Train Symmetry Classification Network')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model to save checkpoints and plots.')
parser.add_argument('--mask', type=str, default='None', help='Type of input masking (Either None, Segmentation, or LAE)')
parser.add_argument('--n_splits', type=int, default=3, help='Number of splits for k-fold')


args = parser.parse_args()

# Check if CUDA (GPU support) is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Hyperparameters
# learning_rate = 0.0001
# learning_rate = 7.288363176209835e-05
# weight_decay = 0.000354797637684891
# BATCH_SIZE = 8
# use_weighted_loss = False

# learning_rate = 2.3211710721108277e-05
# weight_decay = 0.000961722259069207
# BATCH_SIZE = 32
# use_weighted_loss = True

learning_rate = 3.3444923951034694e-05
weight_decay = 0.0009669302289843405
BATCH_SIZE = 8
use_weighted_loss = True

num_epochs = 100
patience = 10  # For early stopping
INPUT_SIZE = '256, 256'
h, w = map(int, INPUT_SIZE.split(','))
NUM_CLASSES_CLS = 2
MODEL_NAME = args.model_name
n_splits = args.n_splits


# Load Datasets
if args.mask == 'None':
    datadir = '../../data/SymDerm_v2/SymDerm_extended/'
    
if args.mask == 'Segmentation':
    datadir = '../../data/SymDerm_v2/SymDerm_mask/'

else:
    datadir = '../../data/SymDerm_v2/SymDerm_lae_mask/'
    
train_csv = 'data_files/SymDerm_v2_train.csv'
test_csv = 'data_files/SymDerm_v2_test.csv'
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)
labels = train_data['labels_symmetry'].values

# Initialize k-fold cross-validation
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
# Initialize variables to store performance metrics for each fold
val_precision, val_recall, val_f1, val_bacc, val_kappa = [], [], [], [], []
test_precision, test_recall, test_f1, test_bacc, test_kappa = [], [], [], [], []


for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f'Fold {fold+1}/{n_splits}')
    
    # Split the data into training and validation sets
    train_subset = Subset(train_symmetry_data(datadir, train_data, crop_size=(h, w), mask=args.mask), train_idx)
    val_subset = Subset(val_symmetry_data(datadir, train_data, crop_size=(h, w), mask=args.mask), val_idx)
    
    # Create DataLoaders for each subset
    trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # Initialize or reset the model, loss function, and optimizer for each fold
    model = models.squeezenet1_0(pretrained=False)
    model.load_state_dict(torch.load('models/squeezenet1_0-b66bff10.pth', map_location=device))
    model.classifier[1] = nn.Conv2d(512, NUM_CLASSES_CLS, kernel_size=(1,1), stride=(1,1))
    model.num_classes = NUM_CLASSES_CLS
    model = model.to(device)

    
    if use_weighted_loss:
        label_counts = train_data['labels_symmetry'].value_counts()
        weights = 1.0 / label_counts
        weights_tensor = torch.tensor(weights.sort_index().values, dtype=torch.float)
        loss_function = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    else:
        loss_function = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Early stopping setup
    best_loss = float('inf')
    epochs_no_improve = 0

    # For plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

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
                loss = loss_function(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct_predictions += (predicted == labels).sum().item()
                val_total_predictions += labels.size(0)

        val_loss = val_running_loss / len(valloader)
        val_accuracy = val_correct_predictions / val_total_predictions
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Fold {fold+1} Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        path = 'models/'+ MODEL_NAME
        if not os.path.isdir(path):
            os.mkdir(path)
        # Early stopping and saving the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, os.path.join(path, f'symmetry_net_fold_{fold+1}_best.pth'))
            print('Saving Model!!!')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered at epoch {epoch+1}!!! Best weights saved for epoch {epoch+1-patience}')
                break
            else:
                print(f'Weights last saved at epoch {epoch+1-epochs_no_improve}')

        # Plotting and saving plots
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Fold {fold+1} Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title(f'Fold {fold+1} Accuracies')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(path, f'Fold_{fold+1}_plot.png'))
        plt.close()


    MODEL_PATH = os.path.join(path, f'symmetry_net_fold_{fold+1}_best.pth')
    testloader = DataLoader(val_symmetry_data(datadir, test_data, crop_size=(h, w), mask=args.mask), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # Assuming the model is already loaded and moved to the correct device
    model = torch.load(MODEL_PATH).to(device)
    model.eval()

    # Containers for true labels and predictions
    all_labels = []
    all_preds = []

    print(f'Fold {fold+1}: Evaluation on Validation Set')
    with torch.no_grad():
        for inputs, labels, names in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass and get predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    conf_mat = confusion_matrix(all_labels, all_preds)
    b_acc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    print(f'Confusion Matrix:\n{conf_mat}')
    print(f'Balanced Accuracy (B.Acc): {b_acc:.4f}')
    print(f'Kappa Score: {kappa:.4f}')
    print('Classification Report (Weighted Average reval_recall, Recall, F1):')
    print(classification_report(all_labels, all_preds, zero_division=0))

    # To get weighted average of precision, recall, and F1 score individually
    precision_weighted = class_report['weighted avg']['precision']
    recall_weighted = class_report['weighted avg']['recall']
    f1_weighted = class_report['weighted avg']['f1-score']

    val_precision.append(precision_weighted)
    val_recall.append(recall_weighted)
    val_f1.append(f1_weighted)
    val_bacc.append(b_acc)
    val_kappa.append(kappa)

    print(f'Weighted Average Precision: {precision_weighted:.4f}')
    print(f'Weighted Average Recall: {recall_weighted:.4f}')
    print(f'Weighted Average F1 Score: {f1_weighted:.4f}')

    # Containers for true labels and predictions
    all_labels = []
    all_preds = []

    print(f'Fold {fold+1}: Evaluation on Test Set')
    with torch.no_grad():
        for inputs, labels, names in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass and get predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate metrics
    conf_mat = confusion_matrix(all_labels, all_preds)
    b_acc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    print(f'Confusion Matrix:\n{conf_mat}')
    print(f'Balanced Accuracy (B.Acc): {b_acc:.4f}')
    print(f'Kappa Score: {kappa:.4f}')
    print('Classification Report (Weighted Average Precision, Recall, F1):')
    print(classification_report(all_labels, all_preds, zero_division=0))

    # To get weighted average of precision, recall, and F1 score individually
    precision_weighted = class_report['weighted avg']['precision']
    recall_weighted = class_report['weighted avg']['recall']
    f1_weighted = class_report['weighted avg']['f1-score']

    test_precision.append(precision_weighted)
    test_recall.append(recall_weighted)
    test_f1.append(f1_weighted)
    test_bacc.append(b_acc)
    test_kappa.append(kappa)
    print(f'Weighted Average Precision: {precision_weighted:.4f}')
    print(f'Weighted Average Recall: {recall_weighted:.4f}')
    print(f'Weighted Average F1 Score: {f1_weighted:.4f}')


print(f'Cross Validation Results for {n_splits} folds:')
print(f'Balanced Accuracy = {np.mean(val_bacc):.3f} \u00B1 {np.std(val_bacc):.3f}')
print(f'Kappa Score = {np.mean(val_kappa):.3f} \u00B1 {np.std(val_kappa):.3f}')
print(f'Weighted Average Precision = {np.mean(val_precision):.3f} \u00B1 {np.std(val_precision):.3f}')
print(f'Weighted Average Recall = {np.mean(val_recall):.3f} \u00B1 {np.std(val_recall):.3f}')
print(f'Weighted Average F1-Score = {np.mean(val_f1):.3f} \u00B1 {np.std(val_f1):.3f}')
print('\n')
print(f'Test Set results for {n_splits} folds:')
print(f'Balanced Accuracy = {np.mean(test_bacc):.3f} \u00B1 {np.std(test_bacc):.3f}')
print(f'Kappa Score = {np.mean(test_kappa):.3f} \u00B1 {np.std(test_kappa):.3f}')
print(f'Weighted Average Precision = {np.mean(test_precision):.3f} \u00B1 {np.std(test_precision):.3f}')
print(f'Weighted Average Recall = {np.mean(test_recall):.3f} \u00B1 {np.std(test_recall):.3f}')
print(f'Weighted Average F1-Score = {np.mean(test_f1):.3f} \u00B1 {np.std(test_f1):.3f}')