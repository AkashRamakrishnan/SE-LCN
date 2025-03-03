import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
from net.models import DynamicCNNModel, SymNet, Xception_dilation
from datasets import train_symmetry_data, val_symmetry_data, val_classification_data_4c
from torch.utils import data
from torchvision import models
import numpy as np

# MODEL_PATH = 'models/classifier_isic_1/MaskCN_e16.pth'  # Replace XXX with the epoch number of the saved model
MODEL_PATH = 'models/classification_efficientnet_4c_trial4/MaskCN_e20.pth'  # Replace XXX with the epoch number of the saved model
INPUT_SIZE = '256, 256'
NUM_CLASSES = 3
BATCH_SIZE = 1


model = torch.load(MODEL_PATH)
print("Model loaded successfully.")

# datadir = '../../data/SymDerm_v2/SymDerm_extended/'
# test_csv = 'data_files/SymDerm_v2_val.csv'
# data_val_list = 'dataset/ISIC/Validation_crop9_cls.txt'
# datadir = '../MB-DCNN/dataset/data/ISIC-2017_Validation_Data/Images/'
# test_csv = '../MB-DCNN/dataset/data/ISIC-2017_Validation_Part3_GroundTruth.csv'
datadir = '../../data/HAM10000_inpaint/'
test_csv = 'data_files/HAM10000_metadata_test.csv'
maskdir = '../../data/HAM10000_lae_masks'
test_loader = data.DataLoader(val_classification_data_4c(datadir, maskdir, test_csv, crop_size=(256, 256)), batch_size=1, shuffle=False,
                            num_workers=8,
                            pin_memory=True)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Make sure the model is in evaluation mode
model.eval()

# Lists to store true and predicted labels
true_labels = []
pred_labels = []

correct_predictions = {class_name: 0 for class_name in ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']}
total_instances = {class_name: 0 for class_name in ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']}

# No gradient is needed for evaluation
with torch.no_grad():
    for inputs, labels, name in test_loader:
        # Transfer to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        
        # Get the predictions
        _, preds = torch.max(outputs, 1)
        
        # Save predictions and true labels
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

        for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            class_name = list(correct_predictions.keys())[label]  # Get class name from label
            total_instances[class_name] += 1
            if label == pred:
                correct_predictions[class_name] += 1

# Compute the metrics
accuracy = accuracy_score(true_labels, pred_labels)
balanced_acc = balanced_accuracy_score(true_labels, pred_labels)
kappa = cohen_kappa_score(true_labels, pred_labels)
# classification_rep = classification_report(true_labels, pred_labels, target_names=['Class1', 'Class2', 'Class3'], output_dict=True)
classification_rep = classification_report(true_labels, pred_labels, target_names=['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df'], output_dict=True)

print(classification_report(true_labels, pred_labels, target_names=['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']))
# Get weighted average metrics
precision_weighted = classification_rep['weighted avg']['precision']
recall_weighted = classification_rep['weighted avg']['recall']
f1_weighted = classification_rep['weighted avg']['f1-score']

class_accuracies = {}
for class_name in correct_predictions:
    class_accuracies[class_name] = correct_predictions[class_name] / total_instances[class_name] if total_instances[class_name] > 0 else 0
    print(f'Accuracy for {class_name}: {class_accuracies[class_name]:.4f}')

# Compute the confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
sensitivity = np.diag(cm) / np.sum(cm, axis=1)
specificity = np.diag(cm) / np.sum(cm, axis=0)

# Compute average sensitivity and specificity
avg_sensitivity = np.mean(sensitivity)
avg_specificity = np.mean(specificity)

# Print the metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Balanced Accuracy: {balanced_acc:.4f}')
print(f'Kappa Score: {kappa:.4f}')
print(f'Weighted Precision: {precision_weighted:.4f}')
print(f'Weighted Recall: {recall_weighted:.4f}')
print(f'Weighted F1-Score: {f1_weighted:.4f}')
print(f'Average Specificity: {avg_specificity:.4f}')
print(f'Average Sensitivity: {avg_sensitivity:.4f}')
