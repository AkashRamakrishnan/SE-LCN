import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

lossTotal = []
lossDice = []
lossRank = []
Jaccard = []

vacc = []
vdice = []
vsensitivity = []
vspecificity = []
vjac = []

outfile = 'slurm-274652.out'
NAME = 'Segmentation_UNET_1/'
path = 'plots/' + NAME
if not os.path.isdir(path):
    os.mkdir(path)

with open(outfile) as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('train_epoch'):
            values = line.split()
            lossTotal.append(float(values[1].split('=')[1][:-1]))
            lossDice.append(float(values[2].split('=')[1][:-1]))
            lossRank.append(float(values[3].split('=')[1][:-1]))
            Jaccard.append(float(values[4].split('=')[1][:-1]))
        elif line.startswith('val'):
            values = line.split()
            vacc.append(float(values[1].split('=')[1][:-1]))
            vdice.append(float(values[2].split('=')[1][:-1]))
            vsensitivity.append(float(values[3].split('=')[1][:-1]))
            vspecificity.append(float(values[4].split('=')[1][:-1]))
            vjac.append(float(values[5].split('=')[1][:-1]))

plt.figure()
plt.plot(lossTotal, label='Total Loss', color='blue', linestyle='--')
plt.legend(loc='best')
plt.savefig(os.path.join(path, 'lossTotal.png'))

plt.figure()
plt.plot(lossDice, label='Dice Loss', color='blue', linestyle='--')
plt.legend(loc='best')
plt.savefig(os.path.join(path, 'lossDice.png'))

plt.figure()
plt.plot(lossRank, label='Rank Loss', color='blue', linestyle='--')
plt.legend(loc='best')
plt.savefig(os.path.join(path, 'lossRank.png'))

plt.figure()
plt.plot(Jaccard, label='Train Jaccard', color='blue', linestyle='--')
plt.plot(vjac, label='Validation Jaccard', color='red', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Jaccard Index')
plt.legend(loc='best')
plt.savefig(os.path.join(path, 'Jaccard.png'))

plt.figure()
plt.plot(vacc, label='Validation Accuracy', color='blue', linestyle='--')
plt.legend(loc='best')
plt.savefig(os.path.join(path, 'vacc.png'))

plt.figure()
plt.plot(vdice, label='Validation Dice', color='blue', linestyle='--')
plt.legend(loc='best')
plt.savefig(os.path.join(path, 'vdice.png'))

plt.figure()
plt.plot(vsensitivity, label='Validation Sensitivity', color='blue', linestyle='--')
plt.legend(loc='best')
plt.savefig(os.path.join(path, 'vsensitivity.png'))

plt.figure()
plt.plot(vspecificity, label='Validation Specificity', color='blue', linestyle='--')
plt.legend(loc='best')
plt.savefig(os.path.join(path, 'vspecificity.png'))

# plt.figure()
# plt.plot(vjac, label='Validation Jaccard', color='blue', linestyle='--')
# plt.legend(loc='best')
# plt.savefig(os.path.join(path, 'vjac.png'))