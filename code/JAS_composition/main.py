import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score

from models.u_model import NIMA
# from models.e_model import NIMA
# from models.relic_model import NIMA
# from models.relic1_model import NIMA
# from models.relic2_model import NIMA
from dataset import JASDataset
from util import AverageMeter
import option

opt = option.init()
opt.device = torch.device("cuda:{}".format(opt.gpu_id))

def adjust_learning_rate(params, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = params.init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_data_part(opt):
    train_data = JASDataset(opt.path_to_train_csv,opt.path_to_imgs)
    test_data = JASDataset(opt.path_to_test_csv,opt.path_to_imgs)
    val_data = JASDataset(opt.path_to_val_csv,opt.path_to_imgs)

    train_loader = DataLoader(train_data,batch_size=opt.batch_size,num_workers= opt.num_workers,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=opt.batch_size,num_workers= opt.num_workers,shuffle=False)
    val_loader = DataLoader(val_data,batch_size=opt.batch_size,num_workers= opt.num_workers,shuffle=False)

    return train_loader,test_loader,val_loader


def train(opt,model, loader, optimizer, criterion, writer=None, global_step=None, name=None):
    model.train()
    train_losses = AverageMeter()
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(opt.device)
        y = y.to(opt.device)
        y_pred = model(x)
        y_pred = y_pred.squeeze(1)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), x.size(0))

        if writer is not None:
            writer.add_scalar(f"{name}/train_loss.avg", train_losses.avg, global_step=global_step + idx)
    return train_losses.avg


def validate(opt,model, loader, criterion, writer=None, global_step=None, name=None):
    model.eval()
    validate_losses = AverageMeter()
    true_score = []
    pred_score = []
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(opt.device)
        y = y.type(torch.FloatTensor)
        y = y.to(opt.device)

        y_pred = model(x)
        y_pred = y_pred.squeeze(1)

        loss = criterion(y_pred, y)
        validate_losses.update(loss.item(), x.size(0))

        if writer is not None:
            writer.add_scalar(f"{name}/val_loss.avg", validate_losses.avg, global_step=global_step + idx)

        for i in range(y.size(0)):
            true_score.append(y[i].data.cpu())
            pred_score.append(y_pred[i].data.cpu())

    lcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)
    print('lcc_mean', lcc_mean[0])
    print('srcc_mean', srcc_mean[0])

    true_score = np.array(true_score)
    true_score_lable = np.where(true_score <= 0.5, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_lable = np.where(pred_score <= 0.5, 0, 1)
    acc = accuracy_score(true_score_lable, pred_score_lable)
    print(acc)

    return validate_losses.avg, acc, lcc_mean, srcc_mean


def start_train(opt):
    train_loader, test_loader, val_loader = create_data_part(opt)
    model = NIMA()
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.init_lr)
    criterion = nn.MSELoss()
    model = model.to(opt.device)
    criterion.to(opt.device)

    writer = SummaryWriter(log_dir=os.path.join(opt.experiment_dir_name, 'logs'))

    for e in range(opt.num_epoch):
        adjust_learning_rate(opt, optimizer, e)

        train_loss = train(opt,model=model, loader=train_loader, optimizer=optimizer, criterion=criterion,
                           writer=writer, global_step=len(train_loader) * e,
                           name=f"{opt.experiment_dir_name}_by_batch")
        val_loss,vacc,vlcc,vsrcc = validate(opt,model=model, loader=val_loader, criterion=criterion,
                            writer=writer, global_step=len(val_loader) * e,
                            name=f"{opt.experiment_dir_name}_by_batch")
        test_loss, tacc, tlcc, tsrcc = validate(opt, model=model, loader=test_loader, criterion=criterion,
                                               writer=writer, global_step=len(test_loader) * e,
                                               name=f"{opt.experiment_dir_name}_by_batch")
        model_name = f"epoch_{e}_.pth"
        torch.save(model.state_dict(), os.path.join(opt.experiment_dir_name, model_name))

        writer.add_scalars("epoch_loss", {'train': train_loss, 'val': val_loss, 'test': test_loss},
                           global_step=e)

        writer.add_scalars("lcc_srcc", {'val_lcc': vlcc[0], 'val_srcc': vsrcc[0],
                                        'test_lcc': tlcc[0], 'test_srcc': tsrcc[0]},
                           global_step=e)

        writer.add_scalars("acc",{'val_acc': vacc, 'test_acc': tacc}, global_step=e)

    writer.close()



def start_check_model(opt):
    _,test_loader, val_loader = create_data_part(opt)
    model = NIMA()
    model.eval()
    model.load_state_dict(torch.load(opt.path_to_model_weight))
    criterion = nn.MSELoss()

    model = model.to(opt.device)
    criterion.to(opt.device)

    test_loss, acc, lcc_mean, srcc_mean = validate(opt,model=model, loader=test_loader, criterion=criterion)
    val_loss, vacc, vlcc_mean, vsrcc_mean = validate(opt,model=model, loader=val_loader, criterion=criterion)

    print('loss:', test_loss, 'acc:', acc, 'lcc:', lcc_mean[0], 'srcc:', srcc_mean[0])
    print('vloss:', val_loss, 'vacc:', vacc, 'vlcc:', vlcc_mean[0], 'vsrcc:', vsrcc_mean[0])

if __name__ =="__main__":

    ### train model
    start_train(opt)
    ### test model
    # start_check_model(opt)