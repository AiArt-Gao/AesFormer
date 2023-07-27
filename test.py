import os
import torch.nn.functional as F
from thop import profile
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
from models.aesformer import Swin_Bert_vlmo_clip, Swin_Bert_vlmo_clip_mean_score, Swin_Bert_vlmo_clip_mean_score_multi_features

from dataset import AVA_Comment_Dataset_bert
from util import AverageMeter, EDMLoss_r1, compute_mae_rmse, set_up_seed
import option
import warnings
warnings.filterwarnings('ignore')

opt = option.init()
opt.device = 'cuda:0' if torch.cuda.is_available() else "cpu"
opt.batch_size = 16


def get_score(opt, y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(opt.device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def cal_flop(model, type='img'):
    if type == 'img':
        img = torch.randn(1, 3, 224, 224).to(opt.device)
        model.eval()
        with torch.no_grad():
            flops, params = profile(model, inputs=(img.unsqueeze(0)))

    elif type == 'text':
        text = 'i love you'
        model.eval()
        with torch.no_grad():
            flops, params = profile(model, inputs=(text, ))
    elif type == 'both':
        img = torch.randn(1, 3, 224, 224).to(opt.device)
        text = 'i love you'
        model.eval()
        with torch.no_grad():
            flops, params = profile(model, inputs=(img, text,))
    print(f"flops:{flops / 1e9}, params:{params / 1e6}")


def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_save_csv, 'train.csv')
    test_csv_path = os.path.join(opt.path_to_save_csv, 'test.csv')
    # AVA_Comment_Dataset_bert
    train_ds = AVA_Comment_Dataset_bert(train_csv_path, opt.path_to_images, if_train=True)
    test_ds = AVA_Comment_Dataset_bert(test_csv_path, opt.path_to_images, if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, test_loader
    # return train_loader, val_loader, test_loader


@torch.no_grad()
def validate(opt, epoch, model, loader, criterion, writer=None, global_step=None, name=None):
    model.eval()
    validate_losses = AverageMeter()
    true_score = []
    pred_score = []
    loader = tqdm(loader)
    for idx, (img, text, y) in enumerate(loader):
        img = img.to(opt.device)
        # y = y.type(torch.FloatTensor)
        y = y.to(opt.device)

        # y_pred = model.train_first_stage(img, text)
        img_pred, text_pred = model.train_first_stage(img, text)
        alpha = 0
        y_pred = alpha * img_pred + (1 - alpha) * text_pred
        # y_pred = model.train_second_stage_with_multi_features(img)
        # y_pred = F.softmax(y_pred, dim=1)
        pscore, pscore_np = get_score(opt, y_pred)
        tscore, tscore_np = get_score(opt, y)

        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()

        loss, _ = criterion(p_target=y, p_estimate=y_pred)
        validate_losses.update(loss.item(), img.size(0))

        loader.desc = "[test epoch {}] loss: {:.4f}".format(epoch, validate_losses.avg)

        if writer is not None:
            writer.add_scalar(f"{name}/val_loss.avg", validate_losses.avg, global_step=global_step + idx)

    plcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)
    # print('lcc_mean:', lcc_mean[0])
    # print('srcc_mean:', srcc_mean[0])

    true_score = np.array(true_score)
    true_score_label = np.where(true_score <= 5.00, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_label = np.where(pred_score <= 5.00, 0, 1)
    acc = accuracy_score(true_score_label, pred_score_label)
    MAE, RMSE = compute_mae_rmse(true_score, pred_score)
    print(f'acc: {acc:.4f}, plcc_mean: {plcc_mean[0]:.3f}, srcc_mean: {srcc_mean[0]:.3f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, EMD: {validate_losses.avg:.4f}')
# acc: 0.8630, plcc_mean: 0.859, srcc_mean: 0.850, MAE: 0.3165, RMSE: 0.4067, EMD: 0.0358


def start_check_model(opt):
    _, test_loader = create_data_part(opt)

    type = 'both'
    model = Swin_Bert_vlmo_clip_mean_score(device=opt.device, depth=2, model_type='tiny', type=type).to(
        opt.device)

    d = torch.load(
        '/data/yuhao/Aesthetics_Quality_Assessment/code/AVA_comment/checkpoint/VLMo/swin_bert_vlmo_clip/mean_score/head/best_mean_srcc.pth',
        map_location='cpu')

    print(model.load_state_dict(d, strict=False))

    model.eval()


    criterion = EDMLoss_r1()

    model = model.to(opt.device)
    criterion.to(opt.device)

    validate(opt, epoch=0, model=model, loader=test_loader, criterion=criterion)


if __name__ == "__main__":
    #### train model
    # start_train(opt)
    #### test model
    set_up_seed()
    start_check_model(opt)
