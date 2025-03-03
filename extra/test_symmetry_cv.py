import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score, classification_report
from datasets import val_symmetry_data  # Assuming you have a similar test dataset loader as your train/val

INPUT_SIZE = '256, 256'
h, w = map(int, INPUT_SIZE.split(','))


# Load your test dataset
datadir = '../../data/SymDerm_v2/SymDerm_extended/'
test_csv = 'data_files/SymDerm_v2_test.csv'  

# datadir = '../../data/SymDerm_v2/SymDerm_mask/'
# test_csv = 'data_files/SymDerm_mask_test.csv'  


MODEL_PATH = 'models/SymmetryNet_squeezenet_trial4/symmetry_net_ep7.pth'
testloader = DataLoader(val_symmetry_data(datadir, test_csv, crop_size=(h, w)), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Assuming the model is already loaded and moved to the correct device
model = torch.load(MODEL_PATH).to(device)
model.eval()

# Containers for true labels and predictions
all_labels = []
all_preds = []

# No gradient needed for evaluation
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

print(f'Weighted Average Precision: {precision_weighted:.4f}')
print(f'Weighted Average Recall: {recall_weighted:.4f}')
print(f'Weighted Average F1 Score: {f1_weighted:.4f}')