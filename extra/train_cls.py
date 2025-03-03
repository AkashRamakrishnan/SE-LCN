import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from sklearn import metrics
from net.models import Xception_dilation, SymNet, DynamicCNNModel
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datasets import train_classification_data, val_classification_data, train_classification_coordinates, val_classification_coordinates
from torch.utils import data
from apex import amp
import torchvision.models as models
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description='Train a neural network model on image data.')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model to save checkpoints and plots.')
parser.add_argument('--mask', type=bool, default=False, help='If using masked input')

args = parser.parse_args()


model_urls = {'Xception_dilation': 'models/xception-43020ad28.pth'}

INPUT_SIZE = '256, 256'
h, w = map(int, INPUT_SIZE.split(','))
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0005
INPUT_CHANNEL = 3
# NUM_CLASSES_SEG = 2
EPOCH = 100
NUM_CLASSES_CLS = 7
BATCH_SIZE = 8
STEPS = 50001
FP16 = False
patience = 20  # For early stopping
NAME = args.model_name


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE, i_iter, STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def cla_evaluate(label, binary_score, pro_score):
    acc = metrics.accuracy_score(label, binary_score)
    AP = metrics.average_precision_score(label, pro_score)
    try:
        auc = metrics.roc_auc_score(label, pro_score)
    except ValueError:
        pass
    CM = metrics.confusion_matrix(label, binary_score)
    print(CM)
    sens = float(CM[1, 1]) / float(CM[1, 1] + CM[1, 0])
    spec = float(CM[0, 0]) / float(CM[0, 0] + CM[0, 1])
    return acc, auc, AP, sens, spec


def val_mode_Scls(valloader, model, num):
    # valiadation
    pro_score_crop = []
    label_val_crop = []
    for index, batch in enumerate(valloader):
        data, label, name = batch
        data = data.cuda()
        # coarsemask = coarsemask.unsqueeze(1).cuda()

        model.eval()
        with torch.no_grad():
            # data_cla = torch.cat((data, coarsemask), dim=1)
            pred = model(data)

        pro_score_crop.append(torch.softmax(pred[0], dim=0).cpu().data.numpy())
        label_val_crop.append(label[0].data.numpy())

    pro_score_crop = np.array(pro_score_crop)
    label_val_crop = np.array(label_val_crop)
    # print(pro_score_crop)
    # print(label_val_crop)
    pro_score = []
    label_val = []

    for i in range(int(len(label_val_crop) / num)):
        score_sum = 0
        label_sum = 0
        for j in range(num):
            score_sum += pro_score_crop[i * num + j]
            label_sum += label_val_crop[i * num + j]
        pro_score.append(score_sum / num)
        label_val.append(label_sum / num)

    pro_score = np.array(pro_score)
    binary_score = np.eye(3)[np.argmax(np.array(pro_score), axis=-1)]
    label_val = np.eye(3)[np.int64(np.array(label_val))]
    # m
    label_val_a = label_val[:, 1]
    pro_score_a = pro_score[:, 1]
    binary_score_a = binary_score[:, 1]
    val_acc_m, val_auc_m, val_AP_m, sens_m, spec_m = cla_evaluate(label_val_a, binary_score_a, pro_score_a)
    # sk
    label_val_a = label_val[:, 2]
    pro_score_a = pro_score[:, 2]
    binary_score_a = binary_score[:, 2]
    val_acc_sk, val_auc_sk, val_AP_sk, sens_sk, spec_sk = cla_evaluate(label_val_a, binary_score_a, pro_score_a)

    return val_acc_m, val_auc_m, val_AP_m, sens_m, spec_m, val_acc_sk, val_auc_sk, val_AP_sk, sens_sk, spec_sk

def validate(model, criterion, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # No need to compute gradients during validation
        for images, labels, name in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate average loss and accuracy
    average_loss = total_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100.0

    print(f'Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return average_loss, accuracy


def main():
    """Create the network and start the training."""
    writer = SummaryWriter('models/' + NAME)

    cudnn.enabled = True

    ############# Create mask-guided classification network.

    model = models.efficientnet_b3(pretrained=False)
    model.load_state_dict(torch.load('models/efficientnet_b3_rwightman-cf984f9c.pth'))


    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.cuda()
    if FP16 is True:
        model_cls, optimizer = amp.initialize(model, optimizer, opt_level="O1")


    model.train()
    model.float()

    ce_loss = nn.CrossEntropyLoss()

    cudnn.benchmark = True

    ############# Load training and validation data

    
    if args.mask:
        datadir = '../../data/HAM10000_lae/'
        train_csv = 'data_files/HAM10000_metadata_train.csv'
        val_csv = 'data_files/HAM10000_metadata_val.csv'
        valdir = '../../data/HAM10000_lae/'

    else:
        datadir = '../../data/HAM10000_balanced/'
        train_csv = 'data_files/HAM10000_metadata_balanced.csv'
        valdir = '../../data/HAM10000_inpaint'
        val_csv = 'data_files/HAM10000_metadata_val.csv'

    coords_csv = 'coordinates.csv'
    trainloader = data.DataLoader(train_classification_coordinates(datadir, train_csv, coords_csv, crop_size=(h, w)),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # datadir = '../../data/SymDerm_v2/SymDerm_extended/'


    # data_val_list = 'dataset/ISIC/Validation_crop9_cls.txt'
    # valdir = '../MB-DCNN/dataset/data/ISIC-2017_Validation_Data/Images/'
    # val_csv = '../MB-DCNN/dataset/data/ISIC-2017_Validation_Part3_GroundTruth.csv'    
    valloader = data.DataLoader(val_classification_coordinates(valdir, val_csv, coords_csv, crop_size=(h, w)), batch_size=1, shuffle=False,
                                num_workers=8,
                                pin_memory=True)

    path = 'models/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'outputxx.txt'
    VAL_LOSS = []
    VAL_ACC = []
    TRAIN_LOSS = []
    best_val_loss = 1000
    epochs_no_improve = 0
    for epoch in range(EPOCH):
        val_m = []
        val_sk = []
        val_mean = []

        train_loss = []

        ############# Start the training
        for i_iter, batch in tqdm(enumerate(trainloader)):

            lr = adjust_learning_rate(optimizer, i_iter)
            writer.add_scalar('learning_rate', lr, i_iter)

            images, labels, name = batch
            input_cla = images.cuda()
            # coarsemask = coarsemask.unsqueeze(1).cuda()
            labels = labels.cuda().long()
            # input_cla = torch.cat((images, coarsemask), dim=1)

            optimizer.zero_grad()
            model.train()
            preds = model(input_cla)

            term = ce_loss(preds, labels.long())
            if FP16 is True:
                with amp.scale_loss(term, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                term.backward()
            optimizer.step()

            train_loss.append(term.cpu().data.numpy())
            writer.add_scalar('loss', term.cpu().data.numpy(), i_iter)

            # if (i_iter > 500) & (i_iter % 100 == 0):
                # epoch = int(i_iter / 100)

        print("train_epoch%d: loss=%f\n" % (epoch, np.nanmean(train_loss)))
        TRAIN_LOSS.append(np.nanmean(train_loss))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_loss, val_acc = validate(model, ce_loss, valloader, device)
        VAL_LOSS.append(val_loss)
        VAL_ACC.append(val_acc)
            ############# Start the validation
        # [val_acc_m, val_auc_m, val_AP_m, val_sens_m, val_spec_m, val_acc_sk, val_auc_sk, val_AP_sk, val_sens_sk,
        # val_spec_sk] = val_mode_Scls(valloader, model, 9)
        # line_val_m = "val%d:vacc_m=%f,vauc_m=%f,vAP_m=%f,vsens_m=%f,spec_m=%f \n" % (
        # epoch, val_acc_m, val_auc_m, val_AP_m, val_sens_m, val_spec_m)
        # line_val_sk = "val%d:vacc_sk=%f,vauc_sk=%f,vAP_sk=%f,vsens_sk=%f,vspec_sk=%f \n" % (
        # epoch, val_acc_sk, val_auc_sk, val_AP_sk, val_sens_sk, val_spec_sk)
        # print(line_val_m)
        # print(line_val_sk)
        # f = open(f_path, "a")
        # f.write(line_val_m)
        # f.write(line_val_sk)

        # val_m.append(np.nanmean(val_auc_m))
        # val_sk.append(np.nanmean(val_auc_sk))
        # val_mean.append((np.nanmean(val_auc_m) + np.nanmean(val_auc_sk)) / 2.)

        # ############# Plot val curves
        plt.figure()
        plt.plot(VAL_LOSS, label='val_loss', color='red')
        plt.plot(TRAIN_LOSS, label='train_loss', color='green')
        # plt.plot(val_mean, label='val_mean', color='blue')
        plt.legend(loc='best')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.savefig(os.path.join(path, 'loss.png'))
        plt.clf()
        plt.close()
        # plt.show()

        # plt.close('all')

        # writer.add_scalar('val_auc_m', np.nanmean(val_auc_m), i_iter)
        # writer.add_scalar('val_auc_sk', np.nanmean(val_auc_sk), i_iter)
        # writer.add_scalar('val_auc_mean', (np.nanmean(val_auc_m) + np.nanmean(val_auc_sk)) / 2., i_iter)

        ############# Save network
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, path + 'MaskCN_e' + str(epoch) + '.pth')
            print("Saving the model!!!!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping triggered.')
                break


if __name__ == '__main__':
    main()

