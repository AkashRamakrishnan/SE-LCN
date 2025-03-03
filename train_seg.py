import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from net.models import deeplabv3plus
from sklearn.metrics import accuracy_score
from net import loss
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from apex import amp
from tensorboardX import SummaryWriter
# from dataset.my_datasets import MyDataSet_seg, MyValDataSet_seg
from datasets import train_segmentation_data, val_segmentation_data
from torch.utils import data
import wandb
from unet import UNet


model_urls = {'deeplabv3plus_xception': 'models/deeplabv3plus_xception_VOC2012_epoch46_all.pth'}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_SIZE = '256, 256'
w, h = map(int, INPUT_SIZE.split(','))
LEARNING_RATE = 0.001
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0005
NUM_CLASSES = 2
TRAIN_NUM = 2000
BATCH_SIZE = 32
EPOCH = 500
STEPS = (TRAIN_NUM/BATCH_SIZE)*EPOCH
FP16 = True
NAME = 'Segmentation_Deeplab_3/'


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE, i_iter, STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def val_mode_seg(valloader, model, path, epoch):
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

        if index in [100]:
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
            fig.savefig(path + name[0][:-4] + '_e' + str(epoch) + '.png', dpi=200, bbox_inches='tight')
            ax.cla()
            fig.clf()
            plt.close()

    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score)


def Jaccard(pred_arg, mask):
    pred_arg = np.argmax(pred_arg.cpu().data.numpy(), axis=1)
    mask = mask.cpu().data.numpy()

    y_true_f = mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], order='F')
    y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2], order='F')

    intersection = float(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score


def main():
    """Create the network and start the training."""
    writer = SummaryWriter('models/' + NAME)

    cudnn.enabled = True

    ############# Create coarse segmentation network
    model = deeplabv3plus(num_classes=NUM_CLASSES)
    # model = UNet(n_channels=3, n_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # model.cuda()
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

    ############# Load training and validation data
    data_train_root = '/home/s3075451/'
    data_train_list = 'data_files/HAM10000_seg_train.txt'
    trainloader = data.DataLoader(train_segmentation_data(data_train_root, data_train_list, crop_size=(w, h)),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    data_val_root = '/home/s3075451/'
    data_val_list = 'data_files/HAM10000_seg_val.txt'
    valloader = data.DataLoader(val_segmentation_data(data_val_root, data_val_list), batch_size=1, shuffle=False, num_workers=8,
                                pin_memory=True)

    path = 'models/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'outputxx.txt'

    val_jac = []
    vjac_max = 0
    ############# Start the training
    for epoch in range(EPOCH):

        train_loss_D = []
        train_loss_R = []
        train_loss_total = []
        train_jac = []

        for i_iter, batch in tqdm(enumerate(trainloader)):

            step = (TRAIN_NUM/BATCH_SIZE)*epoch+i_iter

            images, labels, name = batch
            images = images.cuda()
            labels = labels.cuda().squeeze(1)

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, step)

            model.train()
            preds = model(images)

            loss_D, loss_R = DR_loss(preds, labels)
            term = loss_D + 0.05 * loss_R

            if FP16 is True:
                with amp.scale_loss(term, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                term.backward()

            optimizer.step()

            writer.add_scalar('learning_rate', lr, step)
            writer.add_scalar('loss', term.cpu().data.numpy(), step)

            train_loss_D.append(loss_D.cpu().data.numpy())
            train_loss_R.append(loss_R.cpu().data.numpy())
            train_loss_total.append(term.cpu().data.numpy())
            train_jac.append(Jaccard(preds, labels))

        print("train_epoch%d: lossTotal=%f, lossDice=%f, lossRank=%f, Jaccard=%f \n" % (epoch, np.nanmean(train_loss_total), np.nanmean(train_loss_D), np.nanmean(train_loss_R), np.nanmean(train_jac)))
        # wandb.log({"lossTotal":np.nanmean(train_loss_total), "lossDice":  np.nanmean(train_loss_D), "lossRank": np.nanmean(train_loss_R), "Jaccard": np.nanmean(train_jac)})

        ############# Start the validation
        [vacc, vdice, vsen, vspe, vjac_score] = val_mode_seg(valloader, model, path, epoch)
        line_val = "val%d: vacc=%f, vdice=%f, vsensitivity=%f, vspecifity=%f, vjac=%f \n" % \
                (epoch, np.nanmean(vacc), np.nanmean(vdice), np.nanmean(vsen), np.nanmean(vspe),
                    np.nanmean(vjac_score))
        # wandb.log({"vacc":np.nanmean(vacc), "vdice":  np.nanmean(vdice), "vsensitivity": np.nanmean(vsen), "vspecifity": np.nanmean(vspe), "vjac": np.nanmean(vjac_score)})


        print(line_val)
        f = open(f_path, "a")
        f.write(line_val)

        ############# Plot val curve
        val_jac.append(np.nanmean(vjac_score))
        plt.figure()
        plt.plot(val_jac, label='val jaccard', color='blue', linestyle='--')
        plt.legend(loc='best')

        plt.savefig(os.path.join(path, 'jaccard.png'))
        plt.clf()
        plt.close()
        plt.show()

        plt.close('all')

        writer.add_scalar('val_Jaccard', np.nanmean(vjac_score), epoch)

        ############# Save network
        if np.nanmean(vjac_score) > vjac_max:
            vjac_max = np.nanmean(vjac_score)
            print("Saving Model!!!")
            # torch.save(model.state_dict(), path + 'CoarseSN_e' + str(epoch) + '.pth')
            torch.save(model, path + 'CoarseSN_e' + str(epoch) + '.pth')



if __name__ == '__main__':
    main()
