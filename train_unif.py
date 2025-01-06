from __future__ import print_function

import copy
import math
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.optim as optim
from future.backports import OrderedDict
import numpy as np

from models import wideresnet, resnet, densenet
from attacks import pgd
from utils import nn_util, eval_ood_util
from utils.imagenet_loader import ImageNet
from utils.tinyimages_80mn_loader import TinyImages
import utils.tinyimages_80mn_loader as  TinyImages_ATOM_Util

parser = argparse.ArgumentParser(description='Source code of methos with a uniform distribution')
parser.add_argument('--model_name', default='wrn-40-4', help='model name, wrn-40-4 or resnet-34')
parser.add_argument('--dataset', default='cifar10', help='dataset: svhn, cifar10 or cifar100')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--schedule', type=int, nargs='+', default=[75, 90],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--attack_eps', default=8.0, type=float, help='perturbation radius')
parser.add_argument('--id_pgd_step', default=10, type=int, help='perturb number of steps')
parser.add_argument('--id_aa_step', default=20, type=int, help='perturb number of steps')

parser.add_argument('--attack_lr', default=2.0, type=float, help='perturb step size')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_dir', default='dnn_models/cifar/', help='directory of model for saving checkpoint')
parser.add_argument('--topk_cpts', '-s', default=50, type=int, metavar='N', help='save top-k robust checkpoints')
parser.add_argument('--gpuid', type=int, default=0, help='the ID of GPU.')
parser.add_argument('--training_method', default='pair-pgd-ce',
                    help='training method: clean, clean-pgd-ce, pair-pgd-ce, pgd-ce')
parser.add_argument('--bn_type', default='eval', help='type of batch normalization during attack: train or eval')
parser.add_argument('--random_type', default='gussian', help='random type of pgd: uniform or gussian')
parser.add_argument('--attack_test_step', default=20, type=int, help='perturb number of steps in test phase')
parser.add_argument('--resume_epoch', type=int, default=0, metavar='N', help='epoch for resuming training')
parser.add_argument('--always_save_cpt', action='store_true', default=False,
                    help='whether to save each eopch checkpoint')
parser.add_argument('--norm', default='Linf', type=str, choices=['Linf'])
parser.add_argument('--alpha', default=0., type=float, help='total virtual confidence for ID data')
parser.add_argument('--ood_alpha', default=0., type=float, help='total virtual confidence for OOD data')
parser.add_argument('--num_v_classes', default=0, type=int,
                    help='the number of virtual nodes in last the layer of the net')
parser.add_argument('--id_label_noise', action='store_true', default=False,
                    help='adding noise to the "uniform" labels of adv IDs or not')
parser.add_argument('--ood_label_noise', action='store_true', default=False,
                    help='add noise to the "uniform" labels of (adversarial) OODs or not')
parser.add_argument('--dayu_warmup', default=0, type=int, help='warmup epoch for dayu.')
parser.add_argument('--ood_warmup', default=0, type=int, help='warmup epoch for training on out data.')
parser.add_argument('--ood_file', default='~/workspace/datasets/80M_Tiny_Images/tiny_images.bin',
                    help='tiny_images file ptah')
parser.add_argument('--ood_beta', default=1.0, type=float, help='beta for ood_loss')
parser.add_argument('--auxiliary_dataset', default='80m_tiny_images',
                    choices=['80m_tiny_images', 'imagenet', 'none'], type=str, help='which auxiliary dataset to use')
parser.add_argument('--ood_excl_simi', default=0.35, type=float, help='similarity for excluding auxiliary imagenet')
parser.add_argument('--ood_batch_size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--ood_training_method', default='pair-pgd-ce_in',
                    help='out training method: clean, clean-pgd-ce_in, pair-pgd-ce_in, pgd-ce_in'
                         'clean-pgd-oe_in, pair-pgd-oe_in, pgd-oe_in')
parser.add_argument('--ood_attack_eps', default=8.0, type=float, help='attack epsilon')
parser.add_argument('--ood_pgd_step', default=20, type=int, help='number of iterations for searching adv OODs')
parser.add_argument('--ood_aa_step', default=20, type=int, help='number of iterations for searching adv OODs')

parser.add_argument('--mine_ood', action='store_true', default=False, help='whether to mine informative oods')
parser.add_argument('--mine_method', default='clean', help='ood mining method: clean or fgsm')
parser.add_argument('--quantile', default=0.125, type=float, help='quantile')
parser.add_argument('--storage_device', default='cuda', help='device for computing auroc and fpr: cuda or cpu')
parser.add_argument('--save_socre_dir', default='', type=str, help='dir for saving scores')
parser.add_argument('--best_loss', action='store_true', default=False,
                    help='whether to use adv examples with the best loss during training')
parser.add_argument('--mix_pgd_id_ratio', default=0.5, type=float, help='ratio for PGD adv IDs')
parser.add_argument('--mix_pgd_ood_ratio', default=0.5, type=float, help='ratio for PGD adv OODs')
parser.add_argument('--adv_id_warmup', default=60, type=int, help='warmup epoch for training on adv id data.')
parser.add_argument('--adv_ood_warmup', default=-1, type=int, help='warmup epoch for training on adv ood data.')


args = parser.parse_args()

if args.attack_lr >= 1:
    args.attack_lr = args.attack_lr / 255
if args.attack_eps >= 1:
    args.attack_eps = args.attack_eps / 255
if args.ood_attack_eps >= 1:
    args.ood_attack_eps = args.ood_attack_eps / 255

if args.dataset == 'cifar10':
    NUM_IN_CLASSES = 10
    # NUM_EXAMPLES = 50000
elif args.dataset == 'cifar100':
    NUM_IN_CLASSES = 100
    # NUM_EXAMPLES = 50000
elif args.dataset == 'svhn':
    NUM_IN_CLASSES = 10
    # NUM_EXAMPLES = 73257
# elif args.dataset == 'mnist':
#     NUM_IN_CLASSES = 10
# elif args.dataset == 'fashion-mnist':
#     NUM_IN_CLASSES = 10

else:
    raise ValueError('error dataset: {0}'.format(args.dataset))

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

ID_MIX_TRAINING_METHODS = ['clean-pgd-ce', 'clean-fgsm-ce']
ID_PAIR_TRAINING_METHODS = ['pair-pgd-ce', 'pair-apgd-ce', 'pair-mixpgd-ce', 'pair-id-pgd-ce',]
ID_PGD_TRAINING_METHODS = ['pgd-ce']

OOD_MIX_TRAINING_METHODS = ['clean-pgd-ce_in', 'clean-pgd-oe_in', 'clean-pgd-oe', 'clean-pgd-ce_v', 'clean-apgd-oe', 'clean-mixpgd-oe']
OOD_PAIR_TRAINING_METHODS = ['pair-pgd-ce_in', 'pair-pgd-oe_in', 'pair-pgd-oe', 'pair-pgd-ce_v']
OOD_PGD_TRAINING_METHODS = ['pgd-ce_in', 'pgd-oe_in', 'pgd-oe', 'pgd-ce_v']

if args.save_socre_dir == '':
    args.save_socre_dir = args.model_dir
    print('save_socre_dir is not given, I will set it to model_dir: {}'.format(args.model_dir))

# settings
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if use_cuda else "cpu")
storage_device = torch.device(args.storage_device)


def update_topk_cpts(cur_cpt, training_record, always_save_cpt):
    def save_cpt(save_epoch):
        # save_epoch = cur_cpt['epoch']
        path = os.path.join(args.model_dir, '{0}_model_epoch{1}.pt'.format(args.training_method, save_epoch))
        torch.save(cur_cpt['model'].state_dict(), path)
        path = os.path.join(args.model_dir, '{0}_cpt_epoch{1}.pt'.format(args.training_method, save_epoch))
        torch.save(cur_cpt['optimizer'].state_dict(), path)
        path = os.path.join(args.model_dir, '{0}_trd_epoch{1}.pt'.format(args.training_method, save_epoch))
        torch.save(training_record, path)

    def del_cpt(del_epoch):
        if del_epoch > 0:
            path = os.path.join(args.model_dir, '{0}_model_epoch{1}.pt'.format(args.training_method, del_epoch))
            if os.path.exists(path):
                os.remove(path)
            path = os.path.join(args.model_dir, '{0}_cpt_epoch{1}.pt'.format(args.training_method, del_epoch))
            if os.path.exists(path):
                os.remove(path)
            path = os.path.join(args.model_dir, '{0}_trd_epoch{1}.pt'.format(args.training_method, del_epoch))
            if os.path.exists(path):
                os.remove(path)

    def maintain_cpts():
        files = os.listdir(args.model_dir)
        for file in files:
            full_file = os.path.join(args.model_dir, file)
            if os.path.isfile(full_file) and '.pt' in full_file:
                cur_epoch = file.split('.')[-2].split('epoch')[-1]
                cur_epoch = int(cur_epoch)
                if (cur_epoch not in training_record['nat_topk_cpts']['epoch']) and (
                        cur_epoch not in training_record['rob_fpr_topk_cpts']['epoch']):
                    del_cpt(cur_epoch)
            else:
                pass

    # maintain rob. topk acc
    rob_topk_cpts = training_record['rob_topk_cpts']
    cur_adv_acc = cur_cpt['test_adv_id_acc']
    old_len = len(rob_topk_cpts['epoch'])
    for i in range(0, old_len):
        if cur_adv_acc >= rob_topk_cpts['test_adv_id_acc'][i]:
            rob_topk_cpts['test_adv_id_acc'].insert(i, cur_adv_acc)
            rob_topk_cpts['epoch'].insert(i, cur_cpt['epoch'])
            del rob_topk_cpts['test_adv_id_acc'][-1]
            del rob_topk_cpts['epoch'][-1]
            break
    # maintain nat. topk acc
    nat_topk_cpts = training_record['nat_topk_cpts']
    cur_nat_acc = cur_cpt['test_nat_id_acc']
    old_len = len(nat_topk_cpts['epoch'])
    for i in range(0, old_len):
        if cur_nat_acc >= nat_topk_cpts['test_nat_id_acc'][i]:
            nat_topk_cpts['test_nat_id_acc'].insert(i, cur_nat_acc)
            nat_topk_cpts['epoch'].insert(i, cur_cpt['epoch'])
            del nat_topk_cpts['test_nat_id_acc'][-1]
            del nat_topk_cpts['epoch'][-1]
            break
    # maintain rob. fpr topk acc
    rob_fpr_topk_cpts = training_record['rob_fpr_topk_cpts']
    cur_adv_fpr95 = cur_cpt['test_adv_id_fpr95']
    old_len = len(rob_fpr_topk_cpts['epoch'])
    for i in range(0, old_len):
        if cur_adv_fpr95 <= rob_fpr_topk_cpts['test_adv_id_fpr95'][i]:
            rob_fpr_topk_cpts['test_adv_id_fpr95'].insert(i, cur_adv_fpr95)
            rob_fpr_topk_cpts['epoch'].insert(i, cur_cpt['epoch'])
            del rob_fpr_topk_cpts['test_adv_id_fpr95'][-1]
            del rob_fpr_topk_cpts['epoch'][-1]
            break
    is_cur_cpt_saved = False
    epoch = cur_cpt['epoch']
    save_cpt(epoch)
    if always_save_cpt:
        return True

    maintain_cpts()
    if epoch in training_record['rob_fpr_topk_cpts']['epoch'] or epoch in training_record['nat_topk_cpts']['epoch']:
        is_cur_cpt_saved = True
    return is_cur_cpt_saved


def resume(epoch, model, optimizer, training_record, print_cpt_info=True):
    path = os.path.join(args.model_dir, '{0}_model_epoch{1}.pt'.format(args.training_method, epoch))
    model.load_state_dict(torch.load(path))
    path = os.path.join(args.model_dir, '{0}_cpt_epoch{1}.pt'.format(args.training_method, epoch))
    optimizer.load_state_dict(torch.load(path))
    path = os.path.join(args.model_dir, '{0}_trd_epoch{1}.pt'.format(args.training_method, epoch))
    if os.path.exists(path):
        training_record = torch.load(path)
        print('successfully loaded training_record from {}'.format(path))
        if print_cpt_info:
            print('loaded training_record: {}'.format(training_record))
    else:
        print('I find no training_record from', path)
    return model, optimizer, training_record


def get_all_data(test_loader):
    x_test = None
    y_test = None
    for i, data in enumerate(test_loader):
        batch_x, in_y = data
        if x_test is None:
            x_test = batch_x
        else:
            x_test = torch.cat((x_test, batch_x), 0)

        if y_test is None:
            y_test = in_y
        else:
            y_test = torch.cat((y_test, in_y), 0)
    return x_test, y_test


class OODDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        y = self.labels[index]

        return X, y


def select_ood(ood_loader, model, batch_size, num_in_classes, pool_size, ood_dataset_size, quantile,
               mine_method='clean', attack_args=None):
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    offset = np.random.randint(len(ood_loader.dataset))
    while offset >= 0 and offset < 10000:
        offset = np.random.randint(len(ood_loader.dataset))

    ood_loader.dataset.offset = offset

    out_iter = iter(ood_loader)
    print('Start selecting OOD samples...')

    start = time.time()
    # select ood samples
    model.eval()

    all_ood_input = []
    all_ood_conf = []
    for k in range(pool_size):

        try:
            out_set = next(out_iter)
        except StopIteration:
            offset = np.random.randint(len(ood_loader.dataset))
            while offset >= 0 and offset < 10000:
                offset = np.random.randint(len(ood_loader.dataset))
            ood_loader.dataset.offset = offset
            out_iter = iter(ood_loader)
            out_set = next(out_iter)

        input = out_set[0]
        with torch.no_grad():
            output = model(input.to(device))
        output = F.softmax(output, dim=1)
        if mine_method == 'clean':
            conf = torch.max(output[:, :num_in_classes], dim=1)[0]
        elif mine_method == 'fgsm' and attack_args is not None:
            in_pred = torch.max(output[:, :num_in_classes], dim=1)[1]
            # adv_input = pgd.fgsm_attack_targeted(model, input.to(device), None, in_pred.to(device),
            #                                      attack_eps=attack_args.attack_eps, random_init=True,
            #                                      loss_str='pgd-ce', num_in_classes=num_in_classes,
            #                                      num_out_classes=0)
            adv_input = pgd.pgd_attack_ood_misc(model, input.to(device), None, num_in_classes=num_in_classes,
                                                num_out_classes=0, attack_step=1, attack_lr=attack_args['attack_eps'],
                                                attack_eps=attack_args['attack_eps'], loss_str='pgd-ce_in',
                                                data_type=True, best_loss=args.best_loss)
            with torch.no_grad():
                adv_output = F.softmax(model(adv_input), dim=1)
            conf = torch.max(adv_output[:, :num_in_classes], dim=1)[0]
        else:
            raise ValueError(
                'un-supported parameter combination: mine_method:{}, attack_args:{}'.format(mine_method,
                                                                                            attack_args))
        conf = conf.detach().cpu().numpy()

        all_ood_input.append(input)
        all_ood_conf.extend(conf)

    all_ood_input = torch.cat(all_ood_input, 0)[:ood_dataset_size * 4]
    all_ood_conf = np.array(all_ood_conf)[:ood_dataset_size * 4]
    indices = np.argsort(-all_ood_conf)  # large -> small

    if len(all_ood_conf) < ood_dataset_size * 4:
        print('Warning, the pool_size is too small: batch * pool_size should >= ood_dataset_size * 4')

    N = all_ood_input.shape[0]
    selected_indices = indices[int(quantile * N):int(quantile * N) + ood_dataset_size]

    print('Total OOD samples: ', len(all_ood_conf))
    print('Max in-conf: ', np.max(all_ood_conf), 'Min in-Conf: ', np.min(all_ood_conf), 'Average in-conf: ',
          np.mean(all_ood_conf))

    selected_ood_conf = all_ood_conf[selected_indices]
    print('Selected OOD samples: ', len(selected_ood_conf))
    print('Selected Max in-conf: ', np.max(selected_ood_conf), 'Selected Min conf: ', np.min(selected_ood_conf),
          'Selected Average in-conf: ', np.mean(selected_ood_conf))

    ood_images = all_ood_input[selected_indices]
    # ood_labels = (torch.ones(ood_dataset_size) * output.size(1)).long()
    ood_labels = torch.zeros((ood_images.size(0),)).long().to(ood_images.device)

    ood_train_loader = torch.utils.data.DataLoader(
        OODDataset(ood_images, ood_labels),
        batch_size=batch_size, shuffle=True)
    print('Time: ', time.time() - start)

    return ood_train_loader


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    epoch_lr = args.lr
    for i in range(0, len(args.schedule)):
        if epoch > args.schedule[i]:
            epoch_lr = args.lr * np.power(args.gamma, (i + 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch_lr
    return epoch_lr


def add_noise_to_uniform(unif):
    assert len(unif.size()) == 2
    unif_elem = unif.float().mean()
    new_unif = unif.clone()
    new_unif.uniform_(unif_elem - 0.005 * unif_elem, unif_elem + 0.005 * unif_elem)
    factor = new_unif.sum(dim=1) / unif.sum(dim=1)
    new_unif = new_unif / factor.unsqueeze(dim=1)
    # sum=new_unif.sum(dim=1)
    return new_unif


def constuct_dayu_label(true_label, alpha, in_classes, v_classes, null_classes=0, noise=False):
    # one-hot
    if alpha == 0:
        return F.one_hot(true_label, num_classes=in_classes + v_classes + null_classes).to(true_label.device)
    # standard label smoothing
    elif alpha > 0 and v_classes == 0:
        y_ls = F.one_hot(true_label, num_classes=in_classes) * (1 - alpha) + alpha / in_classes
        y_soft = torch.zeros((len(true_label), in_classes + null_classes))
        y_soft[:, :in_classes] = y_ls
        y_soft = y_soft.to(true_label.device)
        return y_soft
    # with virtual sharing classes
    elif alpha != 0 and v_classes != 0:
        y_vc = torch.zeros((len(true_label), in_classes + v_classes))
        indcs = [i for i in range(len(true_label))]
        y_vc[indcs, true_label] += (1 - alpha)
        temp_v_conf = alpha / v_classes
        if noise:
            y_vc[:, in_classes:in_classes + v_classes] = temp_v_conf
            y_vc[:, in_classes:in_classes + v_classes] = add_noise_to_uniform(y_vc[:, in_classes:in_classes + v_classes])
        else:
            y_vc[:, in_classes:in_classes + v_classes] = temp_v_conf

        y_soft = torch.zeros((len(true_label), in_classes + v_classes + null_classes))
        y_soft[:, :in_classes + v_classes] = y_vc
        y_soft = y_soft.to(true_label.device)
        return y_soft
    else:
        raise ValueError('error alpha:{0} or v_classes:{1}'.format(alpha, v_classes))


def cal_cls_results(logits, y, in_classes, v_classes, data_type='in'):
    msps, preds = torch.max(F.softmax(logits, dim=1)[:, :in_classes], dim=1)
    if data_type == 'in':
        corr_indcs = preds == y
        corr = corr_indcs.sum().item()
        corr_probs = msps[corr_indcs]
        global_preds = torch.max(F.softmax(logits, dim=1), dim=1)[1]
        located_v_indcs = torch.logical_and(global_preds >= in_classes, global_preds < in_classes + v_classes)
        located_v_cls = located_v_indcs.sum().item()
        located_v_corr_indcs = torch.logical_and(corr_indcs, located_v_indcs)
        located_v_corr_cls = located_v_corr_indcs.sum().item()
        return corr, located_v_cls, located_v_corr_cls, msps, corr_probs
    elif data_type == 'out':
        global_preds = torch.max(F.softmax(logits, dim=1), dim=1)[1]
        located_v_cls = torch.logical_and(global_preds >= in_classes, global_preds < in_classes + v_classes).sum().item()
        return located_v_cls, msps
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))


def get_out_uniform_labels(num_input, alpha, num_in_classes, v_classes, noise=False, null_classes=0):
    y_out_soft = torch.zeros((num_input, num_in_classes + v_classes + null_classes))
    if v_classes == 0:
        y_out_soft[:, :num_in_classes + v_classes] = (1 / (num_in_classes + v_classes))
        if noise:
            y_out_soft[:, :num_in_classes + v_classes] = add_noise_to_uniform(y_out_soft[:, :num_in_classes + v_classes])
        return y_out_soft

    # indexes = [i for i in range(len(input))]
    in_conf = (1 - alpha)
    y_out_soft[:, :num_in_classes] = in_conf / num_in_classes
    if noise:
        y_out_soft[:, 0:num_in_classes] = add_noise_to_uniform(y_out_soft[:, 0:num_in_classes])
    v_conf = 1 - in_conf
    if v_classes != 0:
        y_out_soft[:, num_in_classes:num_in_classes + v_classes] = v_conf / v_classes
        if noise:
            y_out_soft[:, num_in_classes:num_in_classes + v_classes] \
                = add_noise_to_uniform(y_out_soft[:, num_in_classes:num_in_classes + v_classes])
    # print(y_out_soft[0:5].to(storage_device).numpy())
    return y_out_soft


def kl_loss(nat_logits, adv_logits):
    batch_size = nat_logits.size()[0]
    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    kl_loss = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(nat_logits, dim=1))
    return kl_loss


def train(model, train_loader, optimizer, test_loader, train_ood_loader, normalizer):
    def get_in_y_soft(in_y, epoch, args):
        if epoch > args.dayu_warmup:
            processed_id_y_soft = constuct_dayu_label(in_y, args.alpha, NUM_IN_CLASSES, args.num_v_classes, args.id_label_noise)
        else:
            processed_id_y_soft = constuct_dayu_label(in_y, 0, NUM_IN_CLASSES, args.num_v_classes, args.id_label_noise)
        return processed_id_y_soft

    def get_out_y_soft(num_x, epoch, args):
        # if epoch > args.dayu_warmup:
        processed_ood_y_soft = get_out_uniform_labels(num_x, args.ood_alpha, NUM_IN_CLASSES, args.num_v_classes, args.ood_label_noise)
        # else:
        #     processed_ood_y_soft = get_out_uniform_labels(num_x, 0, NUM_IN_CLASSES, args.num_v_classes, args.ood_label_noise)
        return processed_ood_y_soft


    def re_process_in_x(model, org_id_x, org_id_y, epoch, args):
        id_training_method = args.training_method
        processed_id_y_soft = get_in_y_soft(org_id_y, epoch, args).to(org_id_y.device)
        if epoch <= args.adv_id_warmup or id_training_method == 'clean':
            return org_id_x, processed_id_y_soft, len(org_id_x)
        loss_str = ''
        if id_training_method in ID_MIX_TRAINING_METHODS:
            len_adv_id = int(len(org_id_x) * 0.5)
            len_nat_id = len(org_id_x) - len_adv_id
            if id_training_method == 'clean-pgd-ce':
                adv_in_x = pgd.pgd_attack(model, org_id_x[-len_adv_id:], org_id_y[-len_adv_id:],
                                          attack_step=args.id_pgd_step, attack_lr=args.attack_lr,
                                          attack_eps=args.attack_eps, random_type=args.random_type,
                                          bn_type=args.bn_type, loss_str='ce', best_loss=args.best_loss)
            else:
                raise ValueError('un-supportted id_training_method: {}'.format(id_training_method))
            advin_y_soft = get_out_y_soft(len_adv_id, epoch, args).to(adv_in_x.device)
            processed_id_x = torch.cat((org_id_x[:len_nat_id], adv_in_x), dim=0)
            processed_id_y_soft[-len_adv_id:] = advin_y_soft
            return processed_id_x, processed_id_y_soft, len_nat_id
        elif id_training_method in ID_PAIR_TRAINING_METHODS:
            len_adv_id = int(len(org_id_x) * 1.0)
            if 'apgd' in id_training_method:
                if id_training_method in ['pair-apgd-ce']:
                    loss_str = 'apgd-ce'
                elif id_training_method in ['pair-apgd-adp-ce_in-out']:
                    loss_str = 'adp-ce_in-out'
                else:
                    raise ValueError('un-supportted id_training_method: {}'.format(id_training_method))
                adv_in_x = pgd.apgd_attack(model, org_id_x[-len_adv_id:], org_id_y[-len_adv_id:],
                                           num_in_classes=NUM_IN_CLASSES, num_out_classes=0,
                                           num_v_classes=args.num_v_classes, attack_step=args.id_aa_step,
                                           attack_eps=args.attack_eps, loss_str=loss_str)
            elif 'mixpgd' in id_training_method:
                len_pgd_id = int(len_adv_id * args.mix_pgd_id_ratio)
                len_apgd_id = len_adv_id - len_pgd_id
                if id_training_method in ['pair-mixpgd-ce']:
                    pgd_loss_str = 'ce'
                    apgd_loss_str = 'apgd-ce'
                else:
                    raise ValueError('un-supportted id_training_method: {}'.format(id_training_method))
                pgd_in_x = pgd.pgd_attack(model, org_id_x[-len_adv_id:-len_pgd_id], org_id_y[-len_adv_id:-len_pgd_id],
                                           num_in_classes=NUM_IN_CLASSES, num_out_classes=0,
                                           num_v_classes=args.num_v_classes, attack_step=args.id_pgd_step,
                                           attack_eps=args.attack_eps, loss_str=pgd_loss_str)
                apgd_in_x = pgd.apgd_attack(model, org_id_x[-len_apgd_id:], org_id_y[-len_apgd_id:],
                                            num_in_classes=NUM_IN_CLASSES, num_out_classes=0,
                                            num_v_classes=args.num_v_classes, attack_step=args.id_aa_step,
                                            attack_eps=args.attack_eps, loss_str=apgd_loss_str)
                adv_in_x=torch.cat((pgd_in_x, apgd_in_x), dim=0)
            else:
                if id_training_method in ['pair-pgd-ce', 'pair-id-pgd-ce']:
                    loss_str = 'ce'
                else:
                    raise ValueError('un-supportted id_training_method: {}'.format(id_training_method))
                adv_in_x = pgd.pgd_attack(model, org_id_x[-len_adv_id:], org_id_y[-len_adv_id:],
                                          attack_step=args.id_pgd_step, attack_lr=args.attack_lr,
                                          attack_eps=args.attack_eps, random_type=args.random_type,
                                          bn_type=args.bn_type, loss_str=loss_str)
            if id_training_method == 'pair-id-pgd-ce':
                advin_y_soft = processed_id_y_soft.detach().clone()
            else:
                advin_y_soft = get_out_y_soft(len_adv_id, epoch, args).to(adv_in_x.device)
            processed_id_x = torch.cat((org_id_x, adv_in_x), dim=0)
            processed_id_y_soft = torch.cat((processed_id_y_soft, advin_y_soft), dim=0)
            return processed_id_x, processed_id_y_soft, len(org_id_x)
        elif id_training_method in ID_PGD_TRAINING_METHODS:
            if id_training_method in ['pgd-ce']:
                adv_in_x = pgd.pgd_attack(model, org_id_x, org_id_y, attack_step=args.id_pgd_step,
                                          attack_lr=args.attack_lr, attack_eps=args.attack_eps,
                                          random_type=args.random_type, bn_type=args.bn_type, loss_str='ce')
            advin_y_soft = get_in_y_soft(org_id_y, epoch, args)
            return adv_in_x, advin_y_soft, 0
        else:
            raise ValueError('unsupported training method: {}'.format(id_training_method))


    def re_process_out_x(model, org_ood_x, epoch, args):
        ood_training_method = args.ood_training_method
        len_org_ood = len(org_ood_x)
        ood_y_soft = get_out_y_soft(len(org_ood_x), epoch, args).to(org_ood_x.device)
        if epoch <= args.adv_ood_warmup:
            return org_ood_x, ood_y_soft, len_org_ood
        if ood_training_method in OOD_MIX_TRAINING_METHODS:
            loss_str=''
            len_adv_ood = int(len_org_ood * 0.5)
            len_nat_ood = len_org_ood - len_adv_ood
            if 'apgd' in ood_training_method:
                if ood_training_method == 'clean-apgd-oe':
                    loss_str = 'apgd-oe'
                    adv_ood_x = pgd.apgd_attack_ood_misc(model, org_ood_x[-len_adv_ood:], None, NUM_IN_CLASSES,
                                                         num_out_classes=0, num_v_classes=args.num_v_classes,
                                                         attack_step=args.ood_aa_step, attack_eps=args.ood_attack_eps,
                                                         loss_str=loss_str)
                else:
                    raise  ValueError('un-supported ood_training_method: {}'.format(ood_training_method))
            elif 'mixpgd' in ood_training_method:
                len_pgd_ood = int(len_adv_ood * args.mix_pgd_ood_ratio)
                len_apgd_ood = len_adv_ood - len_pgd_ood
                y_pgd = ood_y_soft[-len_adv_ood:-len_apgd_ood]
                y_apgd = ood_y_soft[-len_apgd_ood:]
                if ood_training_method == 'clean-mixpgd-oe':
                    pgd_loss_str = 'pgd-oe'
                    apgd_loss_str = 'apgd-oe'
                    y_pgd = None
                    y_apgd = None
                else:
                    raise ValueError('un-supported ood_training_method: {}'.format(ood_training_method))
                pgd_ood_x = pgd.pgd_attack_ood_misc(model, org_ood_x[-len_adv_ood:-len_apgd_ood], y_pgd,
                                                    num_in_classes=NUM_IN_CLASSES, num_out_classes=0,
                                                    attack_step=args.ood_pgd_step, attack_lr=args.attack_lr,
                                                    attack_eps=args.ood_attack_eps, random_type=args.random_type,
                                                    loss_str=pgd_loss_str, best_loss=args.best_loss)
                apgd_ood_x = pgd.apgd_attack_ood_misc(model, org_ood_x[-len_apgd_ood:], y_apgd,
                                                      num_in_classes=NUM_IN_CLASSES, num_out_classes=0, num_v_classes=0,
                                                      attack_step=args.ood_aa_step, attack_eps=args.ood_attack_eps,
                                                      loss_str=apgd_loss_str)
                # print('len of pgd_ood_x: {}, len of apgd_ood_x: {}'.format(len(org_ood_x[-len_adv_id:-len_apgd_id]), len(org_ood_x[-len_apgd_id:])))
                adv_ood_x = torch.cat((pgd_ood_x, apgd_ood_x), dim=0)
            else:
                if ood_training_method == 'clean-pgd-ce_in':
                    loss_str = 'pgd-ce_in'
                elif ood_training_method == 'clean-pgd-oe_in':
                    loss_str = 'pgd-oe_in'
                elif ood_training_method == 'clean-pgd-oe':
                    loss_str = 'pgd-oe'
                elif ood_training_method == 'clean-pgd-ce_v':
                    loss_str = 'pgd-ce_v'
                adv_ood_x = pgd.pgd_attack_ood_misc(model, org_ood_x[-len_adv_ood:], ood_y_soft[-len_adv_ood:],
                                                    NUM_IN_CLASSES, 0, attack_step=args.ood_pgd_step,
                                                    attack_lr=args.attack_lr, attack_eps=args.ood_attack_eps,
                                                    random_init=True, random_type=args.random_type, bn_type='eval',
                                                    loss_str=loss_str, best_loss=args.best_loss)
            processed_ood_x = torch.cat((org_ood_x[:len_nat_ood], adv_ood_x), dim=0)
            return processed_ood_x, ood_y_soft, len_nat_ood
        elif ood_training_method in OOD_PAIR_TRAINING_METHODS:
            loss_str = ''
            if ood_training_method in ['pair-pgd-ce_in']:
                loss_str = 'pgd-ce_in'
            elif ood_training_method in ['pair-pgd-oe_in']:
                loss_str = 'pgd-oe_in'
            elif ood_training_method in ['pair-pgd-oe']:
                loss_str = 'pgd-oe'
            elif ood_training_method in ['pair-pgd-ce_v']:
                loss_str = 'pgd-ce_v'
            len_adv_ood = int(len_org_ood * 1.0)
            adv_ood_x = pgd.pgd_attack_ood_misc(model, org_ood_x[-len_adv_ood:], ood_y_soft[-len_adv_ood:],
                                                NUM_IN_CLASSES, 0, attack_step=args.ood_pgd_step,
                                                attack_lr=args.attack_lr, attack_eps=args.ood_attack_eps,
                                                random_init=True, random_type=args.random_type, bn_type='eval',
                                                loss_str=loss_str, best_loss=args.best_loss)
            processed_ood_x = torch.cat((org_ood_x, adv_ood_x), dim=0)
            processed_ood_y_soft = torch.cat((ood_y_soft, ood_y_soft[-len_adv_ood:].clone()), dim=0)
            return processed_ood_x, processed_ood_y_soft, len(org_ood_x)
        elif ood_training_method in OOD_PGD_TRAINING_METHODS:
            loss_str = ood_training_method
            adv_ood_x = pgd.pgd_attack_ood_misc(model, org_ood_x, ood_y_soft, NUM_IN_CLASSES, 0,
                                                attack_step=args.ood_pgd_step, attack_lr=args.attack_lr,
                                                attack_eps=args.ood_attack_eps, random_init=True,
                                                random_type=args.random_type, bn_type='eval', loss_str=loss_str,
                                                best_loss=args.best_loss)
            return adv_ood_x, ood_y_soft, 0
        elif ood_training_method == 'clean':
            return org_ood_x, ood_y_soft,len(org_ood_x)
        else:
            raise ValueError('unsupported out training method: {0}'.format(ood_training_method))

    training_record = OrderedDict()
    training_record['rob_topk_cpts'] = OrderedDict()
    training_record['rob_topk_cpts']['test_adv_id_acc'] = [-1 for i in range(args.topk_cpts)]
    training_record['rob_topk_cpts']['epoch'] = [0 for i in range(args.topk_cpts)]

    training_record['nat_topk_cpts'] = OrderedDict()
    training_record['nat_topk_cpts']['test_nat_id_acc'] = [-1 for i in range(args.topk_cpts)]
    training_record['nat_topk_cpts']['epoch'] = [0 for i in range(args.topk_cpts)]

    training_record['rob_fpr_topk_cpts'] = OrderedDict()
    training_record['rob_fpr_topk_cpts']['test_adv_id_fpr95'] = [1.1 for i in range(args.topk_cpts)]
    training_record['rob_fpr_topk_cpts']['epoch'] = [0 for i in range(args.topk_cpts)]

    if args.resume_epoch > 0:
        print('try to resume from epoch', args.resume_epoch)
        model, optimizer, training_record = resume(args.resume_epoch, model, optimizer, training_record)

    for epoch in range(args.resume_epoch + 1, args.epochs + 1):
        print('===================================================================================================')
        if train_ood_loader is not None and epoch > args.ood_warmup:
            if args.mine_ood:
                attack_args = {'attack_eps': args.ood_attack_eps}
                num_ood_candidates = len(train_loader.dataset) * math.ceil(args.ood_batch_size / args.batch_size)
                # 2000 * ood_batch_size >= num_ood_candidates * 4
                selected_ood_loader = select_ood(train_ood_loader, model, args.ood_batch_size, NUM_IN_CLASSES,
                                                 pool_size=2000, ood_dataset_size=num_ood_candidates,
                                                 quantile=args.quantile, mine_method=args.mine_method,
                                                 attack_args=attack_args)
                train_ood_iter = enumerate(selected_ood_loader)
            else:
                train_ood_iter = enumerate(train_ood_loader)

        start_time = time.time()
        epoch_lr = adjust_learning_rate(optimizer, epoch)
        # model.train()

        num_nat_ids = 0
        num_adv_ids = 0
        train_nat_id_corr = 0
        train_adv_id_corr = 0
        train_nat_id_msp = torch.tensor([])
        train_adv_id_msp = torch.tensor([])
        
        train_nat_ood_msp = torch.tensor([])
        train_adv_ood_msp = torch.tensor([])
        # train_aa_ood_msp = torch.tensor([])
        for i, data in enumerate(train_loader):
            org_id_x, org_id_y = data
            org_id_x = org_id_x.cuda(non_blocking=True)
            org_id_y = org_id_y.cuda(non_blocking=True)
            processed_id_x, processed_id_y_soft, len_processed_nat_id = re_process_in_x(model, org_id_x, org_id_y, epoch, args)
            # print('processed_id_y_soft.size():', processed_id_y_soft.size(), 'processed_id_y_soft:', processed_id_y_soft[:5])
            # print('processed_id_y_soft.size():', processed_id_y_soft.size(), 'processed_id_y_soft:', processed_id_y_soft[args.batch_size:args.batch_size+5])
            cat_id_x = processed_id_x
            len_cat_id = len(cat_id_x)
            
            if train_ood_loader is not None and epoch > args.ood_warmup:
                _, (org_ood_x, _) = next(train_ood_iter)
                org_ood_x = org_ood_x.cuda(non_blocking=True)
                    
                processed_ood_x, processed_ood_y_soft, len_processed_nat_out = re_process_out_x(model, org_ood_x, epoch, args)
                cat_ood_x = processed_ood_x
                cat_x = torch.cat((cat_id_x, cat_ood_x), dim=0)
                # print('-------------------------------------------------------------------------------------------------')
                # print('processed_ood_y_soft.size():', processed_ood_y_soft.size(), 'processed_ood_y_soft:', processed_ood_y_soft[:5])
                # print('processed_ood_y_soft.size():', processed_ood_y_soft.size(), 'processed_ood_y_soft:', processed_ood_y_soft[args.ood_batch_size:args.ood_batch_size+5])
                # exit()
            else:
                cat_x = processed_id_x
                
            model.train()
            cat_logits = model(cat_x)
            nat_id_loss, adv_id_loss, kl_id_loss = None, None, None
            id_loss = nn_util.cross_entropy_soft_target(cat_logits[:len_cat_id], processed_id_y_soft)

            nat_ood_loss, adv_ood_loss, kl_ood_loss, ood_loss = None, None, None, None
            if epoch > args.ood_warmup:
                ood_loss = nn_util.cross_entropy_soft_target(cat_logits[len_cat_id:], processed_ood_y_soft)
                loss = id_loss + args.ood_beta * ood_loss
            else:
                loss = id_loss

            # compute output
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()

            # statistic training results on IDs
            id_loss = round(id_loss.item(), 6)
            model.eval()
            with torch.no_grad():
                nat_logits = model(org_id_x)
            nat_corr, _, _, nat_id_msp, _ = cal_cls_results(
                nat_logits, org_id_y, NUM_IN_CLASSES, args.num_v_classes, data_type='in')
            train_nat_id_corr += nat_corr
            train_nat_id_msp = torch.cat((train_nat_id_msp, nat_id_msp.cpu()), dim=0)
            if len_cat_id - len_processed_nat_id > 0:
                with torch.no_grad():
                    adv_logits = model(processed_id_x[len_processed_nat_id:len_cat_id])
                if args.training_method in ID_PGD_TRAINING_METHODS:
                    adv_id_corr, _, _, adv_id_msp, _ = cal_cls_results(adv_logits, org_id_y, NUM_IN_CLASSES,
                                                                       args.num_v_classes, data_type='in')
                    train_adv_id_corr += adv_id_corr
                    num_adv_ids += (len_cat_id - len_processed_nat_id)
                else:
                    _, adv_id_msp = cal_cls_results(adv_logits, None, NUM_IN_CLASSES, args.num_v_classes, data_type='out')
                train_adv_id_msp = torch.cat((train_adv_id_msp, adv_id_msp.cpu()), dim=0)

            # statistic training results on OODs
            if ood_loss is not None:
                ood_loss = round(ood_loss.item(), 6)
                if args.ood_training_method in OOD_MIX_TRAINING_METHODS + OOD_PAIR_TRAINING_METHODS:
                    len_adv_out = len(processed_ood_x) - len_processed_nat_out
                    with torch.no_grad():
                        if len_processed_nat_out > 0:
                            nat_out_logits = model(processed_ood_x[:len_processed_nat_out])
                        else:
                            nat_out_logits = model(org_ood_x)
                        if len_adv_out > 0:
                            adv_out_logits = model(processed_ood_x[len_processed_nat_out:])
                elif args.ood_training_method in OOD_PGD_TRAINING_METHODS:
                    len_processed_nat_out = len(processed_ood_x)
                    len_adv_out = len_processed_nat_out
                    with torch.no_grad():
                        nat_out_logits = model(org_ood_x)
                        adv_out_logits = model(processed_ood_x)
                elif args.ood_training_method == 'clean':
                    len_processed_nat_out = len(processed_ood_x)
                    len_adv_out = 0
                    with torch.no_grad():
                        nat_out_logits = model(org_ood_x)
                else:
                    raise ValueError('un-supported ood_training_method: {}'.format(args.ood_training_method))

                # if len_processed_nat_out > 0:
                _, nat_oo_msp = cal_cls_results(nat_out_logits, None, NUM_IN_CLASSES, args.num_v_classes, data_type='out')
                train_nat_ood_msp = torch.cat((train_nat_ood_msp, nat_oo_msp.cpu()), dim=0)
                if len_adv_out > 0:
                    _, adv_ood_msp = cal_cls_results(adv_out_logits, None, NUM_IN_CLASSES, args.num_v_classes, data_type='out')
                    train_adv_ood_msp = torch.cat((train_adv_ood_msp, adv_ood_msp.cpu()), dim=0)

                # #######################################################################################################
                # if epoch > args.schedule[0]:
                #     from autoattack import AutoAttack
                #     version = 'apgd-cw'
                #     attacks_to_run = [version]
                #     adversary = AutoAttack(model, norm='Linf', verbose=False, eps=args.attack_eps, version=version,
                #                            attacks_to_run=attacks_to_run, num_in_classes=NUM_IN_CLASSES,
                #                            num_out_classes=0, num_v_classes=args.num_v_classes, data_type='out')
                #     aa_out_x = adversary.run_standard_evaluation(org_ood_x, None, bs=args.ood_batch_size,
                #                                                  attack_all_emps=False, best_loss=True)
                #     with torch.no_grad():
                #         aa_out_logits = model(aa_out_x)
                #     _, aa_ood_mmsp = cal_cls_results(aa_out_logits, None, NUM_IN_CLASSES,
                #                                                          args.num_v_classes, data_type='out')
                #     train_aa_ood_msp = torch.cat((train_aa_ood_msp, aa_ood_mmsp.cpu()), dim=0)
                # #######################################################################################################

            num_nat_ids += len(org_id_x)
            if i % args.log_interval == 0 or i >= len(train_loader) - 1:
                processed_ratio = round((i / len(train_loader)) * 100, 2)
                print('Train Epoch: {}, Training progress: {}% [{}/{}], In loss: {}, In nat loss: {}, In adv loss: {}, '
                      'In kl loss: {}, Out loss: {}, Out nat loss: {}, Out adv loss: {}, Out kl loss: {},'
                      .format(epoch, processed_ratio, i, len(train_loader), id_loss, nat_id_loss, adv_id_loss,
                              kl_id_loss, ood_loss, nat_ood_loss, adv_ood_loss, kl_ood_loss))

        train_nat_id_acc = (float(train_nat_id_corr) / num_nat_ids)
        train_adv_id_acc = None
        if num_adv_ids > 0:
            train_adv_id_acc = (float(train_adv_id_corr) / num_adv_ids)
        batch_time = time.time() - start_time

        message = 'Epoch {}, Time {}, LR: {}, ID loss: {}, OOD loss:{}'.format(epoch, batch_time, epoch_lr, id_loss, ood_loss)
        print(message)
        in_message = 'Training on ID: nat acc: {}, mean of nat-msp: {}, adv acc: {}, mean of adv-msp: {}' \
            .format(train_nat_id_acc, train_nat_id_msp.mean().item(), train_adv_id_acc, train_adv_id_msp.mean().item())
        print(in_message)
        out_message = 'Training on OOD: mean of nat-msp: {}, mean of adv-msp: {}'\
            .format(train_nat_ood_msp.mean().item(), train_adv_ood_msp.mean().item())
        print(out_message)
        print('----------------------------------------------------------------')

        # Evaluation
        socre_save_dir = os.path.join(args.model_dir, 'epoch_{}_scores'.format(epoch))
        if not os.path.exists(socre_save_dir):
            os.makedirs(socre_save_dir)
        all_in_full_scores_file = os.path.join(socre_save_dir, 'id_misc_scores.txt')
        # clean acc
        test_nat_id_acc, _, _, test_nat_id_msp, test_nat_id_corr_prob, test_nat_id_v_msp, test_nat_id_v_ssp = \
            nn_util.eval(model, test_loader, NUM_IN_CLASSES, args.num_v_classes,
                         misc_score_file=all_in_full_scores_file)
        test_nat_id_mmsp = test_nat_id_msp.mean().item()
        test_nat_id_corr_mprob = test_nat_id_corr_prob.mean().item()
        print('Tetsing ID nat acc: {}, mean of nat-msp: {}, mean of nat corr-prob: {}'
              .format(test_nat_id_acc, test_nat_id_mmsp, test_nat_id_corr_mprob))
        # pgd-20 acc
        test_adv_id_acc, _, _, test_adv_id_msp, test_adv_id_corr_prob = 0., 0, 0, 0., 0.
        if args.training_method in ID_PGD_TRAINING_METHODS or args.training_method in ['pair-id-pgd-ce']:
            print('----------------------------------------')
            test_adv_id_acc, _, _, test_adv_id_msp, test_adv_id_corr_prob, _, _\
                = pgd.eval_pgdadv(model, test_loader, 20, args.attack_lr, args.attack_eps,
                                  num_in_classes=NUM_IN_CLASSES, attack_other_in=False, num_v_classes=args.num_v_classes,
                                  loss_str='ce', best_loss=args.best_loss)
            test_adv_id_corr_mprob = test_adv_id_corr_prob.mean().item()
            test_adv_id_mmsp = test_adv_id_msp.mean().item()
            print('Under PGD-20 attack, test acc on adv-IDs: {}, mean of adv-msp: {}, mean of adv corr-prob: {}'
                  .format(test_adv_id_acc, test_adv_id_mmsp, test_adv_id_corr_mprob))
        adv_id_auroc = 0.
        adv_id_fpr95 = 1.
        if epoch >= args.schedule[0]:
            # # eval detection performance under adaptive attacks
            # print('----------------------------------------')
            # st = time.time()
            # ver = 'apgd-cw'
            # x_test, y_test, aa_x = eval_ood_util.attack_id(model, test_loader, NUM_IN_CLASSES, args, version=ver,
            #                                                 attack_all_emps=False, best_loss=True, device=device)
            # org_incorr_indcs, succ_pertub_indcs, incorr_indcs \
            #     = eval_ood_util.check_adv_status(model, x_test, aa_x, y_test, NUM_IN_CLASSES, args.batch_size)
            # test_aa_id_acc, _, _, test_aa_id_msp, test_aa_id_corr_prob, _, _ \
            #     = nn_util.eval_from_data(model, aa_x, y_test, args.batch_size, NUM_IN_CLASSES, args.num_v_classes)
            # test_aa_id_corr_mprob = test_aa_id_corr_prob.mean().item()
            # test_aa_id_mmsp = test_aa_id_msp.mean().item()
            # print('Under AA {}, test aa acc on IDs: {}, mean of aa-msp: {}, mean of aa-corr-prob: {}'
            #       .format(ver, test_aa_id_acc, test_aa_id_mmsp, test_aa_id_corr_mprob))
            # for key_ind, ind in {'org_incorr_indcs': org_incorr_indcs, 'succ_pertub_indcs': succ_pertub_indcs,
            #                      'incorr_indcs': incorr_indcs}.items():
            #     aa_auroc = nn_util.auroc(test_nat_id_msp, test_aa_id_msp[ind])
            #     aa_fpr95, _ = nn_util.fpr_at_tprN(test_nat_id_msp, test_aa_id_msp[ind], TPR=95)
            #     aa_tpr95, _ = nn_util.tpr_at_tnrN(test_nat_id_msp, test_aa_id_msp[ind], TNR=95)
            #     print('num of {}: {}, mean of msp: {}, AUROC: {}, FPR@TPR95: {}, TPR@TNR95: {}, evaluation time: {}s'
            #           .format(key_ind, ind.sum().item(), test_aa_id_msp[ind].mean().item(), aa_auroc, aa_fpr95,
            #                   aa_tpr95, time.time() - st))
            #     if key_ind == 'succ_pertub_indcs':
            #         adv_id_auroc = aa_auroc
            #         adv_id_fpr95 = aa_fpr95

            print('----------------------------------------')
            print('training performance on OODs:')
            # for key_odd, ood_msp in {'train_nat_ood_msp': train_nat_ood_msp, 'train_adv_ood_msp': train_adv_ood_msp,
            #                          'train_aa_ood_msp': train_aa_ood_msp}.items():
            for key_odd, ood_msp in {'train_nat_ood_msp': train_nat_ood_msp,
                                     'train_adv_ood_msp': train_adv_ood_msp}.items():
                if len(ood_msp) == 0:
                    continue
                st = time.time()
                aa_ood_auroc = nn_util.auroc(test_nat_id_msp, ood_msp)
                aa_ood_fpr95, _ = nn_util.fpr_at_tprN(test_nat_id_msp, ood_msp, TPR=95)
                aa_ood_tpr95, _ = nn_util.tpr_at_tnrN(test_nat_id_msp, ood_msp, TNR=95)
                et = time.time()
                print('num of {}: {}, mean of msp: {}, AUROC:{}, FPR@TPR95: {}, TPR@TNR95: {}, evaluation time: {}s'
                      .format(key_odd, ood_msp.size(0), ood_msp.mean().item(), aa_ood_auroc, aa_ood_fpr95, aa_ood_tpr95,
                              et - st))

            print('----------------------------------------')
            print('performance on val OOD data:')
            out_datasets = ['places365', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
            if args.num_v_classes > 0:
                id_score_dict = {'in_msp': test_nat_id_msp, 'v_ssp-in_msp': test_nat_id_v_ssp - test_nat_id_msp}
            else:
                id_score_dict = {'in_msp': test_nat_id_msp}
            ood_attack_method_arrs = [['clean'], ['adp-ce_in'], ['apgd-adp-cw_in-targeted']]
            for o_a_ms in ood_attack_method_arrs:
                st = time.time()
                _, inputs_num, _, ood_in_msp_dict, _, _, ood_v_ssp_dict \
                    = eval_ood_util.get_ood_scores(model, NUM_IN_CLASSES, socre_save_dir, out_datasets, args,
                                                   ood_attack_methods=o_a_ms, ood_batch_size=128, attack_all_emps=True,
                                                   best_loss=True, pgd_test_step=20, aa_test_step=100,
                                                   taa_test_step=100, targets=9, device=device,
                                                   num_oods_per_set=128 * 4)
                ood_vssp_inmsp_dict = {}
                for temp_key, _ in ood_v_ssp_dict.items():
                    ood_vssp_inmsp_dict[temp_key] = ood_v_ssp_dict[temp_key] - ood_in_msp_dict[temp_key]

                for sc_key, ood_values in {'in_msp': [ood_in_msp_dict, 'in_msp'],
                                           'v_ssp-in_msp': [ood_vssp_inmsp_dict, 'r_ssp']}.items():
                    if sc_key not in id_score_dict:
                        continue
                    con_f = id_score_dict[sc_key]
                    print('=====> using {} as scoring function =====>'.format(sc_key))
                    conf_t = ood_values[0]
                    scoring_func = ood_values[1]
                    (_, _, _, _), (indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score), (mixing_auc, mixing_fprN, mixing_tprN, mixing_score) \
                        = eval_ood_util.eval_on_signle_ood_dataset(con_f, out_datasets, conf_t, ts=[95],
                                                                   scoring_func=scoring_func,
                                                                   storage_device=storage_device)
                    print('{} attacked OODs: with {} scoring function, indiv_auc:{}, indiv_fprN:{}, '
                          'indiv_tprN:{}, indiv_mean_score:{}'.format(o_a_ms, sc_key, indiv_auc, indiv_fprN, indiv_tprN,
                                                                      indiv_mean_score))
                    print("{} attacked OODs: mixing_auc: {}, mixing_fprN: {}, mixing_tprN: {}, "
                          "mixing_score: {}, eval time: {}s".format(o_a_ms, mixing_auc, mixing_fprN, mixing_tprN,
                                                                    mixing_score, time.time() - st))
        # maintain records
        training_record[epoch] = {'loss': loss, 'train_nat_id_acc': train_nat_id_acc,
                                  'train_adv_id_acc': train_adv_id_acc,
                                  'test_nat_id_acc': test_nat_id_acc, 'test_adv_id_acc': test_adv_id_acc,
                                  'test_adv_id_auc': adv_id_auroc, 'test_adv_id_fpr95': adv_id_fpr95
                                  }
        cur_cpt = {'epoch': epoch, 'test_adv_id_acc': test_adv_id_acc, 'test_nat_id_acc': test_nat_id_acc,
                   'test_adv_id_fpr95': adv_id_fpr95, 'model': model, 'optimizer': optimizer}
        update_topk_cpts(cur_cpt, training_record, args.always_save_cpt)

    return training_record


def get_model(model_name, num_in_classes=10, num_out_classes=0, num_v_classes=0, normalizer=None):
    if model_name == 'wrn-34-10':
        return wideresnet.WideResNet(depth=34, widen_factor=10, normalizer=normalizer, num_in_classes=num_in_classes,
                                     num_out_classes=num_out_classes, num_v_classes=num_v_classes)
    elif model_name == 'wrn-28-10':
        return wideresnet.WideResNet(depth=28, widen_factor=10, normalizer=normalizer, num_in_classes=num_in_classes,
                                     num_out_classes=num_out_classes, num_v_classes=num_v_classes)
    elif model_name == 'wrn-40-4':
        return wideresnet.WideResNet(depth=40, widen_factor=4, normalizer=normalizer, num_in_classes=num_in_classes,
                                     num_out_classes=num_out_classes, num_v_classes=num_v_classes)
    elif model_name == 'resnet-18':
        return resnet.ResNet18(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes,
                               num_v_classes=num_v_classes)
    elif model_name == 'resnet-34':
        return resnet.ResNet34(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes,
                               num_v_classes=num_v_classes)
    elif model_name == 'resnet-50':
        return resnet.ResNet50(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes,
                               num_v_classes=num_v_classes)
    elif model_name == 'densenet':
        return densenet.DenseNet3(100, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=normalizer,
                                  num_in_classes=num_in_classes, num_out_classes=num_out_classes,
                                  num_v_classes=num_v_classes)
    else:
        raise ValueError('un-supported model: {0}', model_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    # setup data loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    transform_train = T.Compose([
        # T.Pad(4, padding_mode='reflect'),
        # T.RandomCrop(32),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    transform_test = T.Compose([T.ToTensor()])
    if args.dataset == 'cifar10':
        normalizer = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        dataloader = torchvision.datasets.CIFAR10
        train_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar10', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar10', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)

    elif args.dataset == 'cifar100':
        normalizer = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        dataloader = torchvision.datasets.CIFAR100
        train_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar100', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar100', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'svhn':
        normalizer = None
        svhn_train = torchvision.datasets.SVHN(root='../datasets/svhn', download=True, transform=T.ToTensor(),
                                               split='train')
        svhn_test = torchvision.datasets.SVHN(root='../datasets/svhn', download=True, transform=T.ToTensor(),
                                              split='test')
        train_loader = torch.utils.data.DataLoader(dataset=svhn_train, batch_size=args.batch_size, shuffle=True,
                                                   **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset=svhn_test, batch_size=args.batch_size, shuffle=False,
                                                  **kwargs)
    else:
        raise ValueError('un-supported dataset: {0}'.format(args.dataset))

    out_transform_train = T.Compose([
        T.ToTensor(),
        T.ToPILImage(),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    out_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.auxiliary_dataset == '80m_tiny_images':
        if args.mine_ood:
            ood_select_size = max(args.ood_batch_size, 400)
            ood_loader = torch.utils.data.DataLoader(
                TinyImages_ATOM_Util.TinyImages(transform=out_transform_train, tiny_file=args.ood_file),
                batch_size=ood_select_size, shuffle=False, **out_kwargs)
        else:
            ood_select_size = args.ood_batch_size
            ood_loader = torch.utils.data.DataLoader(
                TinyImages(transform=out_transform_train, tiny_file=args.ood_file),
                batch_size=ood_select_size, shuffle=False, **out_kwargs)
    elif args.auxiliary_dataset == 'imagenet':
        ood_select_size = args.ood_batch_size
        if args.mine_ood:
            ood_select_size = max(args.ood_batch_size, 400)
        ood_loader = torch.utils.data.DataLoader(
            ImageNet(transform=out_transform_train, id_type=args.dataset, exclude_cifar=True,
                     excl_simi=args.ood_excl_simi, img_size=32, imagenet_dir=args.ood_file), batch_size=ood_select_size,
            shuffle=False, **out_kwargs)
    else:
        ood_loader = None

    cudnn.benchmark = True

    # init model, Net() can be also used here for training
    model = get_model(args.model_name, num_in_classes=NUM_IN_CLASSES, num_out_classes=0, num_v_classes=args.num_v_classes,
                      normalizer=normalizer).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print('========================================================================================================')
    print('args:', args)
    print('========================================================================================================')

    train(model, train_loader, optimizer, test_loader, ood_loader, normalizer=normalizer)


if __name__ == '__main__':
    main()
