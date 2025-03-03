import torch
import numpy as np
import os
from net.models import deeplabv3plus
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datasets import test_segmentation_data, val_segmentation_data
from torch.utils import data
from unet import UNet
from tqdm import tqdm

MODEL_PATH = 'models/Segmentation_Deeplab_4/CoarseSN_e125.pth'  # Replace XXX with the epoch number of the saved model
INPUT_SIZE = '256, 256'
NUM_CLASSES = 2
BATCH_SIZE = 1

w, h = map(int, INPUT_SIZE.split(','))

def test_mode_seg(testloader, model):
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []
    for index, batch in enumerate(tqdm(testloader)):
        data, mask, name = batch
        data = data.cuda()
        mask = mask[0].data.numpy()
        val_mask = np.int64(mask > 0)

        model.eval()
        with torch.no_grad():
            pred = model(data)
        # print("Pred1")
        # print(pred)
        pred = torch.softmax(pred, dim=1).cpu().data.numpy()
        # print("Pred2")
        # print(pred)
        # print("Pred3")
        pred_arg = np.argmax(pred[0], axis=0)
        # print("Pred arg")
        # print(pred_arg)
        # print(np.unique(pred_arg))
        y_true_f = val_mask.reshape(val_mask.shape[0]*val_mask.shape[1], order='F')
        y_pred_f = pred_arg.reshape(pred_arg.shape[0]*pred_arg.shape[1], order='F')

        intersection = float(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
        sen.append(intersection / np.sum(y_true_f))
        intersection0 = float(np.sum((1 - y_true_f) * (1 - y_pred_f)))
        spe.append(intersection0 / np.sum(1 - y_true_f))
        acc.append(accuracy_score(y_true_f, y_pred_f))
        jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))

        if index in [100, 200, 300]:
            fig = plt.figure()
            ax = fig.add_subplot(131)
            ax.imshow(data[0].cpu().data.numpy().transpose(1, 2, 0))
            ax.axis('off')
            ax = fig.add_subplot(132)
            ax.imshow(mask)
            ax.axis('off')
            ax = fig.add_subplot(133)
            ax.imshow(pred_arg)
            ax.axis('off')
            fig.suptitle('RGB image,ground truth mask, predicted mask',fontsize=6)
            fig.savefig('test_' + name[0][:-4] + '_e' + '.png', dpi=200, bbox_inches='tight')
            ax.cla()
            fig.clf()
            plt.close()
        

    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score)


def main():
    # model = deeplabv3plus(num_classes=NUM_CLASSES)
    # model = UNet(n_channels=3, n_classes=NUM_CLASSES)
    # model.cuda()
    # model.eval()

    # Load the trained model
    # model.load_state_dict(torch.load(MODEL_PATH))
    
    # pretrained_dict = torch.load(MODEL_PATH)
    # net_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
    # net_dict.update(pretrained_dict)
    # model.load_state_dict(net_dict)
    model = torch.load(MODEL_PATH)
    print("Model loaded successfully.")
    model.cuda()
    model.eval()
    data_test_root = '/home/s3075451/'
    data_test_list = 'data_files/HAM10000_seg_test.txt'
    # data_test_root = '/home/s3075451/SLA/MB-DCNN/dataset/data/ISIC-2017_Training_Data'
    # data_test_list = '/home/s3075451/SLA/MB-DCNN/dataset/ISIC/Training_seg.txt'
    testloader = data.DataLoader(val_segmentation_data(data_test_root, data_test_list, crop_size=(w, h)), batch_size=1, shuffle=False,
                                 num_workers=8, pin_memory=True)

    [acc, dice, sen, spe, jac_score] = test_mode_seg(testloader, model)

    print("Test Accuracy:", np.nanmean(acc))
    print("Test Dice Coefficient:", np.nanmean(dice))
    print("Test Sensitivity:", np.nanmean(sen))
    print("Test Specificity:", np.nanmean(spe))
    print("Test Jaccard Score:", np.nanmean(jac_score))


if __name__ == '__main__':
    main()
