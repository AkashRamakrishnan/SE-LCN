import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from net.models import deeplabv3plus
from sklearn.metrics import accuracy_score
from net import loss

import matplotlib.pyplot as plt
from apex import amp
from tensorboardX import SummaryWriter
from datasets import train_segmentation_data, val_segmentation_data
from torch.utils import data
import wandb
import optuna
from unet import UNet
import pandas as pd
from sklearn.model_selection import train_test_split


model_urls = {'deeplabv3plus_xception': 'models/deeplabv3plus_xception_VOC2012_epoch46_all.pth'}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_SIZE = '256, 256'
w, h = map(int, INPUT_SIZE.split(','))
WEIGHT_DECAY = 0.0005
NUM_CLASSES = 2
TRAIN_NUM = 2000
EPOCH = 500
FP16 = True
image_dir = '../../data/HAM10000_inpaint'
mask_dir = '../../data/HAM10000_segmentations'
train_csv = 'data_files/HAM10000_metadata.csv'
patience = 40

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def val_mode_seg(valloader, model):
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []
    for index, batch in enumerate(valloader):

        data, mask, name = batch
        data = data.cuda()
        mask = mask[0].data.numpy()
        val_mask = np.int64(mask > 0)
        # print(name)

        model.eval()
        with torch.no_grad():
            pred = model(data)

        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)

        #y_pred
        y_true_f = val_mask.reshape(val_mask.shape[0]*val_mask.shape[1], order='F')
        y_pred_f = pred_arg.reshape(pred_arg.shape[0]*pred_arg.shape[1], order='F')

        intersection = float(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
        sen.append(intersection / np.sum(y_true_f))
        intersection0 = float(np.sum((1 - y_true_f) * (1 - y_pred_f)))
        spe.append(intersection0 / np.sum(1 - y_true_f))
        acc.append(accuracy_score(y_true_f, y_pred_f))
        jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))


    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score)


def Jaccard(pred_arg, mask):
    pred_arg = np.argmax(pred_arg.cpu().data.numpy(), axis=1)
    mask = mask.cpu().data.numpy()

    y_true_f = mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], order='F')
    y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2], order='F')

    intersection = float(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score


def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    lambda_value = trial.suggest_uniform('lambda_value', 0.01, 0.1)

    cudnn.enabled = True
    model = deeplabv3plus(num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    if FP16 is True:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = torch.nn.DataParallel(model)

    ############# Load pretrained weights
    pretrained_dict = torch.load(model_urls['deeplabv3plus_xception'])
    net_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)
    print(len(net_dict))
    print(len(pretrained_dict))

    model.train()
    model.float()

    DR_loss = loss.Fusin_Dice_rank()

    cudnn.benchmark = True

    df = pd.read_csv(train_csv)
    train_data, val_data = train_test_split(df, test_size=0.25, random_state=42)
    train_data, val_data = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    trainloader = data.DataLoader(train_segmentation_data(image_dir, mask_dir, train_data, crop_size=(w, h)),
                                  batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    valloader = data.DataLoader(val_segmentation_data(image_dir, mask_dir, val_data), batch_size=1, shuffle=False, num_workers=8,
                                pin_memory=True)
    
    epochs_no_improve = 0
    best_jaccard = 0
    for epoch in range(EPOCH):

        train_loss_D = []
        train_loss_R = []
        train_loss_total = []
        train_jac = []

        for i_iter, batch in enumerate(tqdm(trainloader)):

            step = (TRAIN_NUM/batch_size)*epoch+i_iter

            images, labels, name = batch
            images = images.cuda()
            labels = labels.cuda().squeeze(1)

            optimizer.zero_grad()
            lr = learning_rate

            model.train()
            preds = model(images)

            loss_D, loss_R = DR_loss(preds, labels)
            term = loss_D + lambda_value * loss_R

            if FP16 is True:
                with amp.scale_loss(term, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                term.backward()

            optimizer.step()

            train_loss_D.append(loss_D.cpu().data.numpy())
            train_loss_R.append(loss_R.cpu().data.numpy())
            train_loss_total.append(term.cpu().data.numpy())
            train_jac.append(Jaccard(preds, labels))

        print("train_epoch%d: lossTotal=%f, lossDice=%f, lossRank=%f, Jaccard=%f \n" % (epoch, np.nanmean(train_loss_total), np.nanmean(train_loss_D), np.nanmean(train_loss_R), np.nanmean(train_jac)))

        [vacc, vdice, vsen, vspe, vjac_score] = val_mode_seg(valloader, model)
        line_val = "val%d: vacc=%f, vdice=%f, vsensitivity=%f, vspecifity=%f, vjac=%f \n" % \
                (epoch, np.nanmean(vacc), np.nanmean(vdice), np.nanmean(vsen), np.nanmean(vspe),
                    np.nanmean(vjac_score))
        print(line_val)
        if vjac_score > best_jaccard:
            best_jaccard = vjac_score
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return best_jaccard
        


def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print('Best trial:', study.best_trial.params)
    print('Best Jaccard index:', study.best_trial.value)
if __name__ == '__main__':
    main()
