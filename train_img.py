import os
import sys
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
from models.aesformer import Swin_Bert_vlmo_clip_mean_score
from dataset import AVA_Comment_Dataset, AVA_Comment_Dataset_bert, AVA_Comment_Dataset_vit_bert
from util import EDMLoss, AverageMeter, set_up_seed, EDMLoss_r1, Balanced_l2_Loss
import option
import warnings
warnings.filterwarnings('ignore')


opt = option.init()
opt.save_path = ''
f = open(f'{opt.save_path}/log_test.txt', 'a')
opt.device = torch.device("cuda:{}".format(1))
opt.type = 'img'
opt.batch_size = 256
opt.lr = 1e-4
opt.epochs = 50

def adjust_learning_rate(params, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = params.init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_score(opt, y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(opt.device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_save_csv, 'train.csv')
    test_csv_path = os.path.join(opt.path_to_save_csv, 'test.csv')

    train_ds = AVA_Comment_Dataset_bert(train_csv_path, opt.path_to_images, if_train=True)
    test_ds = AVA_Comment_Dataset_bert(test_csv_path, opt.path_to_images, if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, test_loader

def train_second_stage(opt, epoch, model, loader, optimizer, criterion, criterion1):
    model.train()
    emd_losses = AverageMeter()
    mse_losses = AverageMeter()
    true_score = []
    pred_score = []
    loader = tqdm(loader)
    # loader = tqdm(loader, file=sys.stdout)
    for idx, (img, text, y) in enumerate(loader):

        img = img.to(opt.device)
        y = y.to(opt.device)

        y_pred = model.train_second_stage(img)
        loss1 = criterion(p_target=y, p_estimate=y_pred)
        loss2 = criterion1(y, y_pred)
        loss = loss1 + loss2 * 10

        pscore, pscore_np = get_score(opt, y_pred)
        tscore, tscore_np = get_score(opt, y)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        emd_losses.update(loss1.item(), img.size(0))
        mse_losses.update(loss2.item(), img.size(0))
        loader.desc = "[train epoch {}] emd: {:.3f}, mse: {:.3f}".format(epoch, emd_losses.avg, mse_losses.avg)

        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()

    plcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)
    true_score = np.array(true_score)
    true_score_label = np.where(true_score <= 5.00, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_label = np.where(pred_score <= 5.00, 0, 1)
    acc = accuracy_score(true_score_label, pred_score_label)
    print(f'lcc_mean: {plcc_mean[0]:.3f}, srcc_mean: {srcc_mean[0]:.3f}, acc: {acc:.3f}')

    return emd_losses.avg

@torch.no_grad()
def test_second_stage(opt, epoch, model, loader, criterion):
    model.eval()
    emd_losses = AverageMeter()
    true_score = []
    pred_score = []
    loader = tqdm(loader)

    for idx, (img, text, y) in enumerate(loader):

        img = img.to(opt.device)
        y = y.to(opt.device)

        y_pred = model.train_second_stage(img)
        loss = criterion(p_target=y, p_estimate=y_pred)

        pscore, pscore_np = get_score(opt, y_pred)
        tscore, tscore_np = get_score(opt, y)

        emd_losses.update(loss.item(), img.size(0))
        loader.desc = "[test epoch {}] emd: {:.3f}".format(epoch, emd_losses.avg)

        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()

    plcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)
    true_score = np.array(true_score)
    true_score_label = np.where(true_score <= 5.00, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_label = np.where(pred_score <= 5.00, 0, 1)
    acc = accuracy_score(true_score_label, pred_score_label)
    print(f'lcc_mean: {plcc_mean[0]:.3f}, srcc_mean: {srcc_mean[0]:.3f}, acc: {acc:.3f}')

    return emd_losses.avg, plcc_mean[0], srcc_mean[0], acc

def full_queue(model, loader):
    loader = tqdm(loader)
    for idx, (img, text, y) in enumerate(loader):
        img = img.to(opt.device)
        y = y.to(opt.device)
        # tscore, tscore_np = get_score(opt, y)
        model.full_queue(img, text)

def start_train(opt):
    train_loader, test_loader = create_data_part(opt)
    type = opt.type
    model = Swin_Bert_vlmo_clip_mean_score(device=opt.device, depth=2, model_type='tiny', type=type).to(opt.device)

    d = torch.load(
        '/data2/yuhao/checkpoint/vit_base_bert_mean_score/both/005/best_mean_plcc.pth',
        map_location='cpu')
    print(model.load_state_dict(d, strict=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.99))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    # scheduler = cosine_scheduler(optimizer, opt.lr, 10000, len(train_loader) * opt.epochs)

    criterion = EDMLoss().to(opt.device)
    criterion1 = torch.nn.MSELoss().to(opt.device)
    # criterion1 = Balanced_l2_Loss(opt.device).to(opt.device)

    best_acc, best_plcc, best_srcc, best_loss = 0, 0, 0, 100
    for e in range(opt.epochs):
        train_loss = train_second_stage(opt, epoch=e, model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, criterion1=criterion1)

        torch.save(model.state_dict(), f'{opt.save_path}/latest.pth')

        test_loss, test_plcc, test_srcc, test_acc = test_second_stage(opt, epoch=e, model=model, loader=test_loader, criterion=criterion)
        scheduler.step()

        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'{opt.save_path}/best_acc.pth')

        if best_plcc < test_plcc:
            best_plcc = test_plcc
            torch.save(model.state_dict(), f'{opt.save_path}/best_plcc.pth')

        if best_srcc < test_srcc:
            best_srcc = test_srcc
            torch.save(model.state_dict(), f'{opt.save_path}/best_srcc.pth')

        f.write('epoch:%d, plcc:%.3f,srcc:%.3f,acc:%.3f, train_loss:%.4f, test_loss:%.4f\r\n'
            % (e, test_plcc, test_srcc, test_acc, train_loss, test_loss))

        f.flush()

    f.close()

if __name__ == "__main__":
    #### train model
    set_up_seed()
    start_train(opt)
    #### test model
    # start_check_model(opt)
