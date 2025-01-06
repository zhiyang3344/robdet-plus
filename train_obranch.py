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

from models import wideresnet, resnet, densenet, resnet_64x64, resnext_64x64
from attacks import pgd
from utils import nn_util, eval_ood_obranch_util
from utils.downsampled_imagenet800_loader import ImageNet800
from utils.downsampled_imagenet_loader import ImageNetEXCIFAR
from utils.tinyimages_80mn_loader import TinyImages, RandomImages
from utils.tiredImageNet import TieredImageNet
from utils.aux_indistribution_loader import AuxInDistributionImages, AuxTiny200Images

parser = argparse.ArgumentParser(description='Source code of RobDet and RobDet+')
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
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 3407)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_dir', default='dnn_models/cifar/', help='directory of model for saving checkpoint')
parser.add_argument('--topk_cpts', '-s', default=50, type=int, metavar='N', help='save top-k robust checkpoints')
parser.add_argument('--gpuid', type=int, default=0, help='the ID of GPU.')
parser.add_argument('--bn_type', default='eval', help='type of batch normalization during attack: train or eval')
parser.add_argument('--random_type', default='uniform', help='random type of pgd: uniform or gussian')
parser.add_argument('--resume_epoch', type=int, default=0, metavar='N', help='epoch for resuming training')
parser.add_argument('--norm', default='Linf', type=str, choices=['Linf', 'L2'])
parser.add_argument('--storage_device', default='cuda', help='device for computing auroc and fpr: cuda or cpu')
parser.add_argument('--save_socre_dir', default='', type=str, help='dir for saving scores')

parser.add_argument('--attack_eps', default=8.0, type=float, help='perturbation radius')
parser.add_argument('--attack_lr', default=2.0, type=float, help='perturb step size')
parser.add_argument('--id_pgd_step', default=10, type=int, help='perturb number of steps')
parser.add_argument('--id_aa_step', default=30, type=int, help='perturb number of steps')
parser.add_argument('--id_y_type', default='one-hot', type=str, help='one-hot or ls')
parser.add_argument('--adv_id_y_type', default='org', type=str, help='osemi, out-max, out-uniform, osemi-ls')
parser.add_argument('--id_clean_ratio', default=0.5, type=float, help='ratio of clean OODs in clean and avd OODs')
parser.add_argument('--id_pgd_ratio', default=1.0, type=float, help='ratio for PGD-generated OODs in adv OODs (pgd+apgd)')
parser.add_argument('--training_method', default='pair-pgd-ce',
                    help='training method: clean, clean-pgd-ce, pair-pgd-ce, clean-pgd-ce_in-out, pair-pgd-ce_in-out')
parser.add_argument('--adv_id_warmup', default=-1, type=int, help='warmup epoch for training on adv id data.')
parser.add_argument('--attack_test_step', default=20, type=int, help='perturb number of steps in test phase')
parser.add_argument('--best_loss', action='store_true', default=False,
                    help='whether to use adv examples with the best loss during training')
parser.add_argument('--unique_factor', default=0.1, type=float, help='')

parser.add_argument('--aux_id_file', default='', help='auxiliary in-distribution file')
parser.add_argument('--aux_id_batch_size', type=int, default=512, metavar='N',
                    help='input batch size for auxiliary in-distribution training (default: 512)')
parser.add_argument('--aux_id_warmup', default=-1, type=int, help='warmup epoch for training on auxiliary id data.')

parser.add_argument('--num_out_classes', default=4, type=int, help='the number of out classes')
parser.add_argument('--ood_batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--ood_attack_eps', default=8.0, type=float, help='attack epsilon')
parser.add_argument('--ood_training_method', default='clean',
                    help='out training method: clean, pair-pgd-ce_out, clean-pgd-ce_out')
parser.add_argument('--ood_y_type', default='osemi', type=str, help='osemi, out-uniform, out-random, pre-trained')
parser.add_argument('--ood_clean_ratio', default=0., type=float, help='ratio of clean OODs in clean and avd OODs')
parser.add_argument('--ood_pgd_ratio', default=1.0, type=float, help='ratio for PGD-generated OODs in adv OODs (pgd+apgd)')
parser.add_argument('--ood_pgd_step', default=10, type=int, help='number of iterations for searching adv OODs')
parser.add_argument('--ood_aa_step', default=30, type=int, help='number of iterations for searching adv OODs')
parser.add_argument('--adv_ood_warmup', default=-1, type=int, help='warmup epoch for training on adv ood data.')
parser.add_argument('--mine_ood', action='store_true', default=False, help='whether to mine informative oods')
parser.add_argument('--quantile', default=0.125, type=float, help='quantile')
parser.add_argument('--ood_warmup', default=-1, type=int, help='warmup epoch for training on out data.')
parser.add_argument('--ood_file', default='../datasets/80M_Tiny_Images/tiny_images.bin',
                    help='tiny_images file ptah')
parser.add_argument('--ood_beta', default=1.0, type=float, help='beta for ood_loss')
parser.add_argument('--auxiliary_dataset', default='80m_tiny_images',
                    choices=['80m_tiny_images', '300k-random-images', 'downsampled-imagenet',
                             'downsampled-imagenet-800', 'tired-imagenet', 'none'], type=str,
                    help='which auxiliary dataset to use')
parser.add_argument('--num_included_ood_classes', default=-1, type=int,
                    help='number of included ood classes in the auxiliary dataset')
parser.add_argument('--fix_ood', action='store_true', default=False, help='whether to fix the included oods')
parser.add_argument('--num_fixed_ood', default=-1, type=int,
                    help='number of included ood samples in the auxiliary dataset')

parser.add_argument('--iadv_multi_stage', action='store_true', default=False,
                    help='Whether to use multi-staged iterative training for adv ID data')
parser.add_argument('--num_i_stage', default=2, type=int, help='stage number for iadv_multi_stage')
parser.add_argument('--o_multi_stage', action='store_true', default=False,
                    help='Whether to use multi-staged iterative training for OOD data')
parser.add_argument('--num_o_stage', default=2, type=int, help='stage number for o_multi_stage')
parser.add_argument('--oadv_multi_stage', action='store_true', default=False,
                    help='Whether to use multi-staged iterative training for adv OOD data')
parser.add_argument('--num_oadv_stage', default=2, type=int, help='stage number for oadv_multi_stage')
parser.add_argument('--delta_mom_model', default=0, type=int, help='Momentum model for generating pseudo labels')
parser.add_argument('--gpus', type=int, nargs='+', default=[], help='gpus.')

parser.add_argument('--label_model_name', default='', help='lebel model name when ood_y_type is set to pre-trained')
parser.add_argument('--num_label_model_classes', default=10, type=int, help='number of classes of the label model')
parser.add_argument('--label_model_file', default='', help='lebel model file when ood_y_type is set to pre-trained')

parser.add_argument('--mine_aux_id', action='store_true', default=False, help='whether to mine auxillary ids')
parser.add_argument('--i_quantile', default=0.125, type=float, help='quantile for mining informative auxillary ids')

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
elif args.dataset == 'tiny-imagenet-200':
    NUM_IN_CLASSES = 200
else:
    raise ValueError('error dataset: {0}'.format(args.dataset))


ID_PAIR_TRAINING_METHODS = ['pair-mixpgd-ce', 'pair-mixpgd-ce_in-out']
ID_CLN_MIXPGD_TRAINING_METHODS = ['clean-mixpgd-ce', 'clean-mixpgd-ce_in-out']

OOD_CLN_MIXPGD_TRAINING_METHODS = ['clean-mixpgd-ce', 'clean-mixpgd-ce_out', 'clean-mixpgd-ce_in-out', 'clean-mixpgd-ce_out-sum']
OOD_PAIR_TRAINING_METHODS = ['pair-mixpgd-ce', 'pair-mixpgd-ce_out', 'pair-mixpgd-ce_in-out', 'pair-mixpgd-ce_out-sum']

if args.save_socre_dir == '':
    args.save_socre_dir = args.model_dir
    print(f"INFO, save_socre_dir is not given, I have set it to {args.model_dir}")

if args.id_clean_ratio == 1:
    assert args.adv_id_y_type == 'org'

OOD_BETA = args.ood_beta
if args.ood_beta < 0:
    OOD_BETA = args.ood_batch_size / (args.batch_size + args.aux_id_batch_size)
    print('INFO, the given args.ood_beta < 0, I have set OOD_BETA to: {}'.format(OOD_BETA))

# settings
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
# torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
# torch.backends.cudnn.deterministic = True
torch.cuda.set_device(int(args.gpuid))
if len(args.gpus) > 1:
    torch.cuda.set_device(int(args.gpus[0]))
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
storage_device = torch.device(args.storage_device)

if args.num_included_ood_classes > 0:
    assert args.num_included_ood_classes >= args.num_out_classes
AUX_OOD_CLASSES = []
OOD_CLS_2_NEW_CLS = {}

mom_model = [-1, None]

label_model=None

def save_cpt(cur_cpt, model_dir, training_method, epoch):
    path = os.path.join(model_dir, '{0}_model_epoch{1}.pt'.format(training_method, epoch))
    torch.save(cur_cpt['model'].state_dict(), path)
    path = os.path.join(model_dir, '{0}_cpt_epoch{1}.pt'.format(training_method, epoch))
    torch.save(cur_cpt['optimizer'].state_dict(), path)
    path = os.path.join(model_dir, '{0}_trd_epoch{1}.pt'.format(training_method, epoch))
    torch.save(cur_cpt['record'], path)


def del_cpt(model_dir, training_method, epoch):
    path = os.path.join(model_dir, '{0}_model_epoch{1}.pt'.format(training_method, epoch))
    if os.path.exists(path):
        os.remove(path)
    path = os.path.join(model_dir, '{0}_cpt_epoch{1}.pt'.format(training_method, epoch))
    if os.path.exists(path):
        os.remove(path)
    path = os.path.join(model_dir, '{0}_trd_epoch{1}.pt'.format(training_method, epoch))
    if os.path.exists(path):
        os.remove(path)


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

def resume_model(epoch, model):
    path = os.path.join(args.model_dir, '{0}_model_epoch{1}.pt'.format(args.training_method, epoch))
    model.load_state_dict(torch.load(path))
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        return model, True
    else:
        print('{} is not exist'.format(path))
        return model, False


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


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    epoch_lr = args.lr
    for i in range(0, len(args.schedule)):
        if epoch > args.schedule[i]:
            epoch_lr = args.lr * np.power(args.gamma, (i + 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch_lr
    return epoch_lr


def cal_cls_results(logits, y, num_in_classes, num_out_classes, data_type='in'):
    output = F.softmax(logits, dim=1)
    conf, pred_indx = torch.max(output, dim=1)
    num_located_out = torch.logical_and(pred_indx >= num_in_classes, pred_indx < num_in_classes + num_out_classes).sum().item()
    in_msp = output[:, :num_in_classes].max(dim=1)[0]
    out_msp = torch.tensor([], device=logits.device)
    out_ssp = torch.tensor([], device=logits.device)
    if num_out_classes > 0:
        out_msp = output[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[0]
        out_ssp = output[:, num_in_classes:num_in_classes + num_out_classes].sum(dim=1)

    if data_type == 'in':
        _, in_pred = torch.max(logits[:, :num_in_classes], dim=1)
        corr = (in_pred == y).sum().item()
        return corr, conf, in_msp, (num_located_out, out_msp, out_ssp)
    elif data_type == 'out':
        return conf, in_msp, (num_located_out, out_msp, out_ssp)
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))


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


def select_ood(ood_loader, model, batch_size, num_in_classes, num_out_classes, num_pool_iters, ood_dataset_size, quantile,
               data_type='out'):
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
    all_ood_labels = []
    for k in range(num_pool_iters):
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
        ood_conf = torch.sum(output[:, num_in_classes:], dim=1)
        ood_conf = ood_conf.detach().cpu().numpy()

        all_ood_input.append(input)
        all_ood_conf.extend(ood_conf)

        _, pred_within_out = torch.max(output[:, num_in_classes:num_in_classes + num_out_classes], dim=1)
        pred_within_out += num_in_classes
        all_ood_labels.append(pred_within_out)

    all_ood_input = torch.cat(all_ood_input, 0)
    all_ood_conf = np.array(all_ood_conf)
    print('Total iterated OOD samples:', len(all_ood_input))

    all_ood_input = all_ood_input[:ood_dataset_size * 4]
    all_ood_conf = all_ood_conf[:ood_dataset_size * 4]
    all_ood_labels = torch.cat(all_ood_labels, 0)[:ood_dataset_size * 4]
    # indices = np.argsort(-all_ood_conf)  # large -> small
    if data_type=='out':
        indices = np.argsort(all_ood_conf)  # small -> large
    elif data_type=='in':
        indices = np.argsort(all_ood_conf)[::-1]  # large -> small
    else:
        raise ValueError('Error, unsupported data_type: {}'.format(data_type))

    if len(all_ood_conf) < ood_dataset_size * 4:
        print('Warning, the num_pool_iters is too small: batch * num_pool_iters should >= ood_dataset_size * 4')

    N = all_ood_input.shape[0]
    selected_indices = indices[int(quantile * N):int(quantile * N) + ood_dataset_size]

    print('Total candidate OOD samples: ', len(all_ood_conf))
    print('Max in-conf: ', np.max(all_ood_conf), 'Min in-conf: ', np.min(all_ood_conf), 'Average in-conf: ',
          np.mean(all_ood_conf))

    selected_ood_conf = all_ood_conf[selected_indices]
    print('Selected OOD samples: ', len(selected_ood_conf))
    print('Selected Max in-conf: ', np.max(selected_ood_conf), 'Selected Min in-conf: ', np.min(selected_ood_conf),
          'Selected Average in-conf: ', np.mean(selected_ood_conf))

    ood_images = all_ood_input[selected_indices]
    # ood_labels = (torch.ones(ood_dataset_size) * output.size(1)).long()
    ood_labels = all_ood_labels[selected_indices]

    ood_train_loader = torch.utils.data.DataLoader(
        OODDataset(ood_images, ood_labels),
        batch_size=batch_size, shuffle=True)
    print('Time: ', time.time() - start)

    return ood_train_loader


def get_semi_random_label(logits, num_in_classes, num_out_classes):
    pseudo_label = torch.zeros((logits.size(0),), device=logits.device).long()
    # assert logits.size(1) == (num_in_classes + num_out_classes)
    whole_pred = torch.max(logits[:, :num_in_classes + num_out_classes], dim=1)[1]
    indcs = whole_pred >= num_in_classes
    pseudo_label[indcs] = whole_pred[indcs]
    # print('before indcs.sum()', indcs.sum(), 'pseudo_label', pseudo_label)
    if (~indcs).sum() > 0:
        random_label = torch.randint(num_in_classes, num_in_classes + num_out_classes, ((~indcs).sum().item(),)).to(logits.device)
        pseudo_label[~indcs] = random_label
    return pseudo_label


def get_out1_label(num_samples, num_in_classes):
    return torch.zeros((num_samples,)).to(torch.int64) + num_in_classes

def get_random_label(num_labels, num_in_classes, num_out_classes):
    random_label = torch.randint(num_in_classes, num_in_classes + num_out_classes, (num_labels, ))
    return random_label

def get_out_uniform_label(size, num_in_classes, num_out_classes):
    assert size[1] == (num_in_classes + num_out_classes)
    out_ls_label = torch.zeros((size[0], (num_in_classes + num_out_classes))).float()
    u = torch.arange(0, size[0])
    out_ls_label[u, num_in_classes:] = 1 / num_out_classes
    return out_ls_label

def kl_loss(nat_logits, adv_logits):
    batch_size = nat_logits.size()[0]
    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    kl_loss = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(nat_logits, dim=1))
    return kl_loss


def train(model, train_loader, optimizer, test_loader, aux_id_train_loader, train_ood_loader, normalizer):
    if args.label_model_name != '':
        if args.label_model_file =='':
            pass
        else:
            label_model = get_model(args.label_model_name, num_in_classes=args.num_label_model_classes,
                                    num_out_classes=0, num_v_classes=0, normalizer=normalizer, dataset=args.dataset)
            label_model.load_state_dict(torch.load(args.label_model_file, map_location='cpu'))
            label_model = label_model.to(device)

    def get_in_y_soft(in_y):
        return F.one_hot(in_y, num_classes=NUM_IN_CLASSES + args.num_out_classes).to(in_y.device)

    def get_out_y_soft(logits=None, ood_y_type='osemi', ood_y=None):
        assert args.num_out_classes > 0
        if ood_y_type == 'osemi':
            assert logits is not None
            temp_out_y = get_semi_random_label(logits, NUM_IN_CLASSES, args.num_out_classes)
            ood_y_soft = F.one_hot(temp_out_y, num_classes=NUM_IN_CLASSES + args.num_out_classes)
        elif ood_y_type == 'out1':
            temp_out_y = get_out1_label(logits.size(0), NUM_IN_CLASSES).to(logits.device)
            ood_y_soft = F.one_hot(temp_out_y, num_classes=NUM_IN_CLASSES + args.num_out_classes)
        elif ood_y_type == 'out-random':
            temp_out_y = get_random_label(logits.size(0), NUM_IN_CLASSES, args.num_out_classes).to(logits.device)
            ood_y_soft = F.one_hot(temp_out_y, num_classes=NUM_IN_CLASSES + args.num_out_classes)
        elif ood_y_type == 'out-uniform':
            ood_y_soft = get_out_uniform_label([logits.size(0), NUM_IN_CLASSES + args.num_out_classes], NUM_IN_CLASSES, args.num_out_classes)
        elif ood_y_type == 'oh':
            assert ood_y is not None
            ood_y_soft = F.one_hot(ood_y, num_classes=NUM_IN_CLASSES + args.num_out_classes)
        elif ood_y_type == 'pre-trained' or ood_y_type == 'self-trained':
            assert logits is not None
            preds = logits.max(dim=1)[1]
            temp_out_y = NUM_IN_CLASSES + preds % args.num_out_classes
            ood_y_soft = F.one_hot(temp_out_y, num_classes=NUM_IN_CLASSES + args.num_out_classes)
        else:
            raise ValueError('un-supported ood_y_type:'.format(ood_y_type))
        return ood_y_soft.to(logits.device)

    def re_process_in_x(model, org_id_x, org_id_y, epoch, id_training_method):
        in_y_soft = get_in_y_soft(org_id_y).to(org_id_x.device)
        if epoch <= args.adv_id_warmup or id_training_method == 'clean':
            return org_id_x, in_y_soft, len(org_id_x)
        elif id_training_method in ID_PAIR_TRAINING_METHODS:
            if args.adv_id_y_type == 'org':
                adv_id_y_soft = in_y_soft
            else:
                with torch.no_grad():
                    if args.adv_id_y_type == 'pre-trained':
                        assert label_model is not None
                        label_model.eval()
                        temp_logits = label_model(org_id_x)
                    else:
                        model.eval()
                        if args.delta_mom_model <= 0:
                            temp_logits = model(org_id_x)
                        else:
                            temp_logits = mom_model[1](org_id_x)
                    adv_id_y_soft = get_out_y_soft(logits=temp_logits, ood_y_type=args.adv_id_y_type)
            len_pgd_id = int(len(org_id_x) * args.id_pgd_ratio)
            len_apgd_id = len(org_id_x) - len_pgd_id
            if id_training_method in ['pair-mixpgd-ce']:
                pgd_loss_str = 'ce'
                apgd_loss_str = 'apgd-ce'
            elif id_training_method in ['pair-mixpgd-ce_in-out']:
                pgd_loss_str = 'ce_in-out'
                apgd_loss_str = 'apgd-adp-ce_in-out'
            else:
                raise ValueError('un-supportted id_training_method: {}'.format(id_training_method))
            pgd_id_x = torch.tensor([], device=org_id_x.device)
            apgd_id_x = torch.tensor([], device=org_id_x.device)
            if len_pgd_id > 0:
                pgd_id_x = pgd.pgd_attack(model, org_id_x[:len_pgd_id], org_id_y[:len_pgd_id],
                                          attack_step=args.id_pgd_step, attack_lr=args.attack_lr,
                                          attack_eps=args.attack_eps, norm=args.norm,
                                          loss_str=pgd_loss_str, num_in_classes=NUM_IN_CLASSES,
                                          num_out_classes=args.num_out_classes, num_v_classes=0,
                                          best_loss=args.best_loss)
            if len_apgd_id > 0:
                apgd_id_x = pgd.apgd_attack(model, org_id_x[len_pgd_id:], org_id_y[len_pgd_id:],
                                            num_in_classes=NUM_IN_CLASSES, num_out_classes=args.num_out_classes,
                                            num_v_classes=0, attack_step=args.id_aa_step, attack_eps=args.attack_eps,
                                            loss_str=apgd_loss_str, norm=args.norm, )
            adv_id_x = torch.cat((pgd_id_x, apgd_id_x), 0)
            id_x = torch.cat((org_id_x, adv_id_x), 0)
            id_y_soft = torch.cat((in_y_soft, adv_id_y_soft), dim=0)
            return id_x, id_y_soft, len(org_id_x)
        elif id_training_method in ID_CLN_MIXPGD_TRAINING_METHODS:
            len_nat_id = int(len(org_id_x) * args.id_clean_ratio)
            len_adv_id = len(org_id_x) - len_nat_id
            if args.adv_id_y_type == 'org':
                adv_id_y_soft = in_y_soft[len_nat_id:]
            else:
                with torch.no_grad():
                    if args.adv_id_y_type == 'pre-trained':
                        assert label_model is not None
                        label_model.eval()
                        temp_logits = label_model(org_id_x[len_nat_id:])
                    else:
                        model.eval()
                        if args.delta_mom_model <= 0:
                            temp_logits = model(org_id_x[len_nat_id:])
                        else:
                            temp_logits = mom_model[1](org_id_x[len_nat_id:])
                    adv_id_y_soft = get_out_y_soft(logits=temp_logits, ood_y_type=args.adv_id_y_type)
            len_pgd_id = int(len_adv_id * args.id_pgd_ratio)
            len_apgd_id = len_adv_id - len_pgd_id
            if id_training_method in ['clean-mixpgd-ce']:
                pgd_loss_str = 'ce'
                apgd_loss_str = 'apgd-ce'
            elif id_training_method in ['clean-mixpgd-ce_in-out']:
                pgd_loss_str = 'ce_in-out'
                apgd_loss_str = 'apgd-adp-ce_in-out'
            else:
                raise ValueError('un-supportted id_training_method: {}'.format(id_training_method))
            pgd_id_x = torch.tensor([], device=org_id_x.device)
            apgd_id_x = torch.tensor([], device=org_id_x.device)
            if len_pgd_id > 0:
                pgd_id_x = pgd.pgd_attack(model, org_id_x[len_nat_id:][:len_pgd_id], org_id_y[len_nat_id:][:len_pgd_id],
                                          attack_step=args.id_pgd_step, attack_lr=args.attack_lr,
                                          attack_eps=args.attack_eps, norm=args.norm,
                                          loss_str=pgd_loss_str, num_in_classes=NUM_IN_CLASSES,
                                          num_out_classes=args.num_out_classes, num_v_classes=0,
                                          best_loss=args.best_loss)
            if len_apgd_id > 0:
                apgd_id_x = pgd.apgd_attack(model, org_id_x[len_nat_id:][len_pgd_id:],
                                            org_id_y[len_nat_id:][len_pgd_id:], num_in_classes=NUM_IN_CLASSES,
                                            num_out_classes=args.num_out_classes, num_v_classes=0,
                                            attack_step=args.id_aa_step, attack_eps=args.attack_eps,
                                            loss_str=apgd_loss_str, norm=args.norm)
            adv_id_x = torch.cat((pgd_id_x, apgd_id_x), 0)

            id_x = torch.cat((org_id_x[:len_nat_id], adv_id_x), 0)
            id_y_soft = torch.cat((in_y_soft[:len_nat_id], adv_id_y_soft), dim=0)
            return id_x, id_y_soft, len_nat_id
        else:
            raise ValueError('unsupported training method: {0}'.format(id_training_method))

    def re_process_out_x(model, org_ood_x, org_ood_y, epoch, ood_training_method, ood_clean_ratio=-1):
        if ood_clean_ratio < 0:
            ood_clean_ratio = args.ood_clean_ratio
        with torch.no_grad():
            model.eval()
            if args.ood_y_type == 'pre-trained':
                assert label_model is not None
                label_model.eval()
                temp_logits = label_model(org_ood_x)
                ood_preds =  model(org_ood_x).max(dim=1)[1].cpu()
                ood_pred_bincount = ood_preds.bincount().cpu()
            else:
                if args.delta_mom_model <= 0:
                    temp_logits = model(org_ood_x)
                else:
                    mom_model[1].eval()
                    temp_logits = mom_model[1](org_ood_x)
                ood_preds = temp_logits.max(dim=1)[1].cpu()
                ood_pred_bincount = ood_preds.bincount().cpu()
            ood_y_soft = get_out_y_soft(logits=temp_logits, ood_y_type=args.ood_y_type, ood_y=org_ood_y)
        if ood_training_method == 'clean' or epoch <= args.adv_ood_warmup:
            return org_ood_x, ood_y_soft, len(org_ood_x), ood_preds, ood_pred_bincount
        elif ood_training_method in OOD_CLN_MIXPGD_TRAINING_METHODS:
            len_nat_ood = int(len(org_ood_x) * ood_clean_ratio)
            len_adv_ood = len(org_ood_x) - len_nat_ood
            len_pgd_ood = int(len_adv_ood * args.ood_pgd_ratio)
            len_apgd_ood = len_adv_ood - len_pgd_ood
            ood_y_hard = ood_y_soft.max(dim=1)[1]
            if ood_training_method == 'clean-mixpgd-ce':
                pgd_loss_str = 'pgd-ce'
                apgd_loss_str = 'apgd-ce'
            elif ood_training_method == 'clean-mixpgd-ce_out':
                pgd_loss_str = 'pgd-ce_out'
                apgd_loss_str = 'apgd-adp-ce_out'
            elif ood_training_method == 'clean-mixpgd-ce_in-out':
                pgd_loss_str = 'pgd-ce_in-out'
                apgd_loss_str = 'apgd-adp-ce_in-out'
            elif ood_training_method == 'clean-mixpgd-ce_out-sum':
                pgd_loss_str = 'pgd-ce_out-sum'
                apgd_loss_str = 'apgd-adp-ce_out-sum'
            else:
                raise ValueError('un-supported training_method: {}'.format(ood_training_method))
            pgd_ood_x = torch.tensor([], device=org_ood_x.device)
            apgd_ood_x = torch.tensor([], device=org_ood_x.device)
            if len_pgd_ood > 0:
                pgd_ood_x = pgd.pgd_attack_ood_misc(model, org_ood_x[len_nat_ood:][:len_pgd_ood],
                                                    ood_y_hard[len_nat_ood:][:len_pgd_ood],
                                                    num_in_classes=NUM_IN_CLASSES, num_out_classes=args.num_out_classes,
                                                    attack_step=args.ood_pgd_step, attack_lr=args.attack_lr,
                                                    attack_eps=args.ood_attack_eps, norm=args.norm,
                                                    loss_str=pgd_loss_str, best_loss=args.best_loss)
            if len_apgd_ood > 0:
                apgd_ood_x = pgd.apgd_attack_ood_misc(model, org_ood_x[len_nat_ood:][len_pgd_ood:],
                                                      ood_y_hard[len_nat_ood:][len_pgd_ood:],
                                                      num_in_classes=NUM_IN_CLASSES,
                                                      num_out_classes=args.num_out_classes, num_v_classes=0,
                                                      attack_step=args.ood_aa_step, attack_eps=args.ood_attack_eps,
                                                      norm=args.norm, loss_str=apgd_loss_str)
            adv_ood_x = torch.cat((pgd_ood_x, apgd_ood_x), dim=0)
            processed_ood_x = torch.cat((org_ood_x[:len_nat_ood], adv_ood_x), dim=0)
            processed_ood_y_soft = ood_y_soft
            return processed_ood_x, processed_ood_y_soft, len_nat_ood, ood_preds, ood_pred_bincount
        elif ood_training_method in OOD_PAIR_TRAINING_METHODS:
            len_pgd_ood = int(len(org_ood_x) * args.ood_pgd_ratio)
            len_apgd_ood = len(org_ood_x) - len_pgd_ood
            ood_y_hard = ood_y_soft.max(dim=1)[1]
            if ood_training_method == 'pair-mixpgd-ce':
                pgd_loss_str = 'pgd-ce'
                apgd_loss_str = 'apgd-ce'
            elif ood_training_method == 'pair-mixpgd-ce_out':
                pgd_loss_str = 'pgd-ce_out'
                apgd_loss_str = 'apgd-adp-ce_out'
            elif ood_training_method == 'pair-mixpgd-ce_in-out':
                pgd_loss_str = 'pgd-ce_in-out'
                apgd_loss_str = 'apgd-adp-ce_in-out'
            elif ood_training_method == 'pair-mixpgd-ce_out-sum':
                pgd_loss_str = 'pgd-ce_out-sum'
                apgd_loss_str = 'apgd-adp-ce_out-sum'
            else:
                raise ValueError('un-supported training_method: {}'.format(ood_training_method))
            pgd_ood_x = torch.tensor([], device=org_ood_x.device)
            apgd_ood_x = torch.tensor([], device=org_ood_x.device)
            if len_pgd_ood > 0:
                pgd_ood_x = pgd.pgd_attack_ood_misc(model, org_ood_x[:len_pgd_ood], ood_y_hard[:len_pgd_ood],
                                                    num_in_classes=NUM_IN_CLASSES, num_out_classes=args.num_out_classes,
                                                    attack_step=args.ood_pgd_step, attack_lr=args.attack_lr,
                                                    attack_eps=args.ood_attack_eps, norm=args.norm,
                                                    loss_str=pgd_loss_str, best_loss=args.best_loss)
            if len_apgd_ood > 0:
                apgd_ood_x = pgd.apgd_attack_ood_misc(model, org_ood_x[len_pgd_ood:], ood_y_hard[len_pgd_ood:],
                                                      num_in_classes=NUM_IN_CLASSES,
                                                      num_out_classes=args.num_out_classes, num_v_classes=0,
                                                      attack_step=args.ood_aa_step, attack_eps=args.ood_attack_eps,
                                                      norm=args.norm, loss_str=apgd_loss_str)
            adv_ood_x = torch.cat((pgd_ood_x, apgd_ood_x), dim=0)
            processed_ood_x = torch.cat((org_ood_x, adv_ood_x), dim=0)
            processed_ood_y_soft = torch.cat((ood_y_soft, ood_y_soft.clone()), dim=0)
            return processed_ood_x, processed_ood_y_soft, len(org_ood_x), torch.cat((ood_preds, ood_preds.clone()), dim=0), torch.cat((ood_pred_bincount, ood_pred_bincount.clone()), dim=0)
        else:
            raise ValueError('unsupported out training method: {0}'.format(ood_training_method))

    training_record = OrderedDict()
    if args.resume_epoch > 0:
        print('try to resume from epoch', args.resume_epoch)
        model, optimizer, training_record = resume(args.resume_epoch, model, optimizer, training_record)

        mom_epoch = args.resume_epoch - args.delta_mom_model
        assert mom_epoch > 0
        rtn_model, flag = resume_model(mom_epoch, copy.deepcopy(model))
        if flag:
            mom_model[1], mom_model[0] = rtn_model, mom_epoch
            print('Info, loaded model of epoch: {} as mom_model'.format(mom_model[0]))
        else:
            print('Error, model of epoch: {} is not exists'.format(mom_epoch))

    for epoch in range(args.resume_epoch + 1, args.epochs + 1):
        if epoch == 1:
            mom_model[1], mom_model[0] = copy.deepcopy(model), 1

        print('===================================================================================================')
        if args.delta_mom_model <= 0:
            print("Info, use the current epoch's model to generate pseudo-labels on the fly.")
        else:
            print('>>>>> Info, model of epoch: {} is mom_model'.format(mom_model[0]))
        if train_ood_loader is not None and epoch > args.ood_warmup:
            if args.mine_ood:
                num_ood_candidates = len(train_loader.dataset) * math.ceil(args.ood_batch_size / args.batch_size)
                # 2000 * ood_batch_size >= num_ood_candidates * 4
                selected_ood_loader = select_ood(train_ood_loader, model, args.ood_batch_size, NUM_IN_CLASSES,
                                                 args.num_out_classes, num_pool_iters=2000,
                                                 ood_dataset_size=num_ood_candidates, quantile=args.quantile)
                train_ood_iter = enumerate(selected_ood_loader)
            else:
                train_ood_iter = enumerate(train_ood_loader)
        aux_id_train_iter = None
        if aux_id_train_loader is not None and epoch > args.aux_id_warmup:
            if args.mine_aux_id:
                num_aux_id_candidates = len(train_loader.dataset) * math.ceil(args.aux_id_batch_size / args.batch_size)
                # 2000 * ood_batch_size >= num_ood_candidates * 4
                selected_aux_id_loader = select_ood(train_ood_loader, model, args.aux_id_batch_size, NUM_IN_CLASSES,
                                                    args.num_out_classes, num_pool_iters=2000,
                                                    ood_dataset_size=num_aux_id_candidates, quantile=args.i_quantile,
                                                    data_type='in')
                aux_id_train_iter = enumerate(selected_aux_id_loader)
            else:
                aux_id_train_iter = enumerate(aux_id_train_loader)

        start_time = time.time()
        epoch_lr = adjust_learning_rate(optimizer, epoch)

        num_nat_ids = 0
        num_adv_ids = 0
        train_nat_id_corr = 0
        train_adv_id_corr = 0
        train_nat_id_msp = torch.tensor([])
        train_adv_id_msp = torch.tensor([])
        train_nat_id_out_msp = torch.tensor([])
        train_adv_id_out_msp = torch.tensor([])
        train_nat_id_out_ssp = torch.tensor([])
        train_adv_id_out_ssp = torch.tensor([])

        train_nat_ood_msp = torch.tensor([])
        train_adv_ood_msp = torch.tensor([])
        train_nat_ood_out_msp = torch.tensor([])
        train_adv_ood_out_msp = torch.tensor([])
        train_nat_ood_out_ssp = torch.tensor([])
        train_adv_ood_out_ssp = torch.tensor([])

        train_num_nat_id_pred_out = 0
        train_num_adv_id_pred_out = 0
        train_num_nat_ood_pred_out = 0
        train_num_adv_ood_pred_out = 0
        num_ood = 0
        all_ood_correct = 0
        all_ood_pred_bincount = torch.zeros((NUM_IN_CLASSES + args.num_out_classes), dtype=torch.int64)
        for i, data in enumerate(train_loader):
            org_id_x, org_id_y = data
            if aux_id_train_iter is not None and epoch > args.aux_id_warmup:
                _, (aux_id_x, aux_id_y) = next(aux_id_train_iter)
                org_id_x = torch.cat((org_id_x, aux_id_x), dim=0)
                org_id_y = torch.cat((org_id_y, aux_id_y), dim=0)

            org_id_x = org_id_x.cuda(non_blocking=True)
            org_id_y = org_id_y.cuda(non_blocking=True)
            if args.iadv_multi_stage and epoch % args.num_i_stage == 1:
                processed_id_x, processed_id_y_soft, len_processed_nat_id = re_process_in_x(model, org_id_x, org_id_y, epoch, 'clean')
            else:
                processed_id_x, processed_id_y_soft, len_processed_nat_id = re_process_in_x(model, org_id_x, org_id_y, epoch, args.training_method)
            cat_id_x = processed_id_x
            len_cat_id = len(cat_id_x)
            # print('processed_id_y_soft.size():', processed_id_y_soft.size(), 'processed_id_y_soft:', processed_id_y_soft[:5])
            # print('processed_id_y_soft.size():', processed_id_y_soft.size(), 'processed_id_y_soft:', processed_id_y_soft[args.batch_size:args.batch_size+5])
            if train_ood_loader is not None and epoch > args.ood_warmup:
                if args.o_multi_stage and epoch % args.num_o_stage == 1:
                    cat_x = cat_id_x
                else:
                    try:
                        if args.auxiliary_dataset == 'tired-imagenet':
                            j, (org_ood_x, org_ood_y_spec, org_ood_y) = next(train_ood_iter)
                        else:
                            j, (org_ood_x, org_ood_y) = next(train_ood_iter)
                    except:
                        del train_ood_iter
                        train_ood_iter = enumerate(train_ood_loader)
                        if args.auxiliary_dataset == 'tired-imagenet':
                            newj, (org_ood_x, org_ood_y_spec, org_ood_y) = next(train_ood_iter)
                        else:
                            newj, (org_ood_x, org_ood_y) = next(train_ood_iter)
                        print('newj:', newj)
                    org_ood_x = org_ood_x.cuda(non_blocking=True)
                    if len(OOD_CLS_2_NEW_CLS) > 0: # and (args.num_included_ood_classes > 0 or args.fix_ood is True):
                        for indx in range(0, len(org_ood_y)):
                            org_ood_y[indx] = OOD_CLS_2_NEW_CLS[org_ood_y[indx].cpu().item()]
                    org_ood_y = org_ood_y.to(torch.int64).cuda(non_blocking=True)
                    if args.oadv_multi_stage and epoch % args.num_oadv_stage == 1:
                        processed_ood_x, processed_ood_y_soft, len_processed_nat_ood, batch_ood_pred, batch_ood_pred_bincount = \
                            re_process_out_x(model, org_ood_x, org_ood_y, epoch, 'clean')
                    else:
                        processed_ood_x, processed_ood_y_soft, len_processed_nat_ood, batch_ood_pred, batch_ood_pred_bincount = \
                            re_process_out_x(model, org_ood_x, org_ood_y, epoch, args.ood_training_method)
                    cat_ood_x = processed_ood_x
                    cat_x = torch.cat((cat_id_x, cat_ood_x), dim=0)
                    # print('-------------------------------------------------------------------------------------------------')
                    # print('len_processed_nat_ood:', len_processed_nat_ood)
                    # print('processed_ood_y_soft.size():', processed_ood_y_soft.size(), 'processed_ood_y_soft:', processed_ood_y_soft[:5])
                    # print('processed_ood_y_soft.size():', processed_ood_y_soft.size(), 'processed_ood_y_soft:', processed_ood_y_soft[args.ood_batch_size:args.ood_batch_size+5])
                    # exit()
                    all_ood_pred_bincount[:len(batch_ood_pred_bincount)] = all_ood_pred_bincount[:len(batch_ood_pred_bincount)] + batch_ood_pred_bincount
                    all_ood_correct = all_ood_correct + (batch_ood_pred == processed_ood_y_soft.max(dim=1)[1].cpu()).sum()
                    num_ood = num_ood + len(processed_ood_x)
            else:
                cat_x = cat_id_x

            model.train()
            cat_logits = model(cat_x)
            id_loss = nn_util.cross_entropy_soft_target(cat_logits[:len_cat_id], processed_id_y_soft)

            ood_loss = None
            if epoch > args.ood_warmup:
                if args.o_multi_stage and epoch % args.num_o_stage == 1:
                    loss = id_loss
                else:
                    ood_loss = nn_util.cross_entropy_soft_target(cat_logits[len_cat_id:], processed_ood_y_soft)
                    loss = id_loss + OOD_BETA * ood_loss
            else:
                loss = id_loss

            # compute output
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()

            model.eval()
            # statistic training results on IDs
            id_loss = round(id_loss.item(), 6)
            with torch.no_grad():
                nat_id_logits = model(org_id_x)
            nat_id_corr, _, nat_id_msp, (num_nat_id_pred_out, nat_id_out_msp, nat_id_out_ssp) = cal_cls_results(
                nat_id_logits, org_id_y, NUM_IN_CLASSES, args.num_out_classes, data_type='in')
            train_nat_id_corr += nat_id_corr
            train_num_nat_id_pred_out += num_nat_id_pred_out
            train_nat_id_msp = torch.cat((train_nat_id_msp, nat_id_msp.cpu()), dim=0)
            train_nat_id_out_ssp = torch.cat((train_nat_id_out_ssp, nat_id_out_ssp.cpu()), dim=0)
            train_nat_id_out_msp = torch.cat((train_nat_id_out_msp, nat_id_out_msp.cpu()), dim=0)
            if len_cat_id - len_processed_nat_id > 0:
                with torch.no_grad():
                    adv_id_logits = model(processed_id_x[len_processed_nat_id:len_cat_id])
                if args.adv_id_y_type == 'org-oh':
                    adv_id_corr, _, adv_id_msp, (num_adv_id_pred_out, adv_id_out_msp, adv_id_out_ssp) = cal_cls_results(
                        adv_id_logits, org_id_y[len_processed_nat_id:len_cat_id], NUM_IN_CLASSES, args.num_out_classes,
                        data_type='in')
                    train_adv_id_corr += adv_id_corr
                    num_adv_ids += (len_cat_id - len_processed_nat_id)
                else:
                    _, adv_id_msp, (num_adv_id_pred_out, adv_id_out_msp, adv_id_out_ssp) = cal_cls_results(
                        adv_id_logits, None, NUM_IN_CLASSES, args.num_out_classes, data_type='out')
                train_num_adv_id_pred_out += num_adv_id_pred_out
                train_adv_id_msp = torch.cat((train_adv_id_msp, adv_id_msp.cpu()), dim=0)
                train_adv_id_out_ssp = torch.cat((train_adv_id_out_ssp, adv_id_out_ssp.cpu()), dim=0)
                train_adv_id_out_msp = torch.cat((train_adv_id_out_msp, adv_id_out_msp.cpu()), dim=0)

            # statistic training results on OODs
            if ood_loss is not None:
                ood_loss = round(ood_loss.item(), 6)
                if args.ood_training_method in OOD_CLN_MIXPGD_TRAINING_METHODS + OOD_PAIR_TRAINING_METHODS:
                    len_adv_ood = len(processed_ood_x) - len_processed_nat_ood
                    with torch.no_grad():
                        if len_processed_nat_ood > 0:
                            nat_ood_logits = model(processed_ood_x[:len_processed_nat_ood])
                        else:
                            nat_ood_logits = model(org_ood_x)
                        if len_adv_ood > 0:
                            adv_ood_logits = model(processed_ood_x[len_processed_nat_ood:])
                elif args.ood_training_method == 'clean':
                    len_processed_nat_ood = len(processed_ood_x)
                    len_adv_ood = 0
                    with torch.no_grad():
                        nat_ood_logits = model(org_ood_x)
                else:
                    raise ValueError('un-supported ood_training_method: {}'.format(args.ood_training_method))

                # if len_processed_nat_ood > 0:
                _, nat_ood_msp, (num_nat_ood_pred_out, nat_ood_out_msp, nat_ood_out_ssp) = cal_cls_results(nat_ood_logits, None, NUM_IN_CLASSES, args.num_out_classes, data_type='out')
                train_nat_ood_msp = torch.cat((train_nat_ood_msp, nat_ood_msp.cpu()), dim=0)
                train_nat_ood_out_msp = torch.cat((train_nat_ood_out_msp, nat_ood_out_msp.cpu()), dim=0)
                train_nat_ood_out_ssp = torch.cat((train_nat_ood_out_ssp, nat_ood_out_ssp.cpu()), dim=0)
                train_num_nat_ood_pred_out += num_nat_ood_pred_out
                if len_adv_ood > 0:
                    _, adv_ood_msp, (num_adv_ood_pred_out, adv_ood_out_msp, adv_ood_out_ssp) = cal_cls_results(adv_ood_logits, None, NUM_IN_CLASSES, args.num_out_classes, data_type='out')
                    train_adv_ood_msp = torch.cat((train_adv_ood_msp, adv_ood_msp.cpu()), dim=0)
                    train_adv_ood_out_msp = torch.cat((train_adv_ood_out_msp, adv_ood_out_msp.cpu()), dim=0)
                    train_adv_ood_out_ssp = torch.cat((train_adv_ood_out_ssp, adv_ood_out_ssp.cpu()), dim=0)
                    train_num_adv_ood_pred_out += num_adv_ood_pred_out

            num_nat_ids += len(org_id_x)
            if i % args.log_interval == 0 or i >= len(train_loader) - 1:
                processed_ratio = round((i / len(train_loader)) * 100, 2)
                print('Train Epoch: {}, Training progress: {}% [{}/{}], In loss: {}, Out loss: {}'
                      .format(epoch, processed_ratio, i, len(train_loader), id_loss, ood_loss))

        train_nat_id_acc = float(train_nat_id_corr) / num_nat_ids
        train_adv_id_acc = None
        if num_adv_ids > 0:
            train_adv_id_acc = (float(train_adv_id_corr) / num_adv_ids)
        batch_time = time.time() - start_time
        message = 'Epoch {}, Time {}, LR: {}, In loss: {}, Out loss:{}'\
            .format(epoch, batch_time, epoch_lr, id_loss, ood_loss)
        print(message)
        in_message = 'Training on ID: ' \
                     'nat acc: {}, nat-pred-out:{}, mean of nat-msp: {}, mean of nat-out-ssp: {}, ' \
                     'adv acc: {}, adv-pred-out:{}, mean of adv-msp: {}, mean of adv-out-ssp: {}, ' \
            .format(train_nat_id_acc, train_num_nat_id_pred_out, train_nat_id_msp.mean().item(), train_nat_id_out_ssp.mean().item(),
                    train_adv_id_acc, train_num_adv_id_pred_out, train_adv_id_msp.mean().item(), train_adv_id_out_ssp.mean().item(),)
        print(in_message)
        out_message = 'Training on OOD: ' \
                      'nat-pred-out:{}, mean of nat-in-msp: {}, mean of nat-out-msp: {}, mean of nat-out-ssp: {}, ' \
                      'adv-pred-out:{}, mean of adv-in-msp: {}, mean of adv-out-msp: {}, mean of adv-out-ssp: {}' \
            .format(train_num_nat_ood_pred_out, train_nat_ood_msp.mean().item(), train_nat_ood_out_msp.mean().item(), train_nat_ood_out_ssp.mean().item(),
                    train_num_adv_ood_pred_out, train_adv_ood_msp.mean().item(), train_adv_ood_out_msp.mean().item(), train_adv_ood_out_ssp.mean().item())
        print(out_message)
        print('prediction distribution:', all_ood_pred_bincount.numpy().tolist())
        if num_ood > 0:
            print('OOD Acc: {}/{} = {}'.format(all_ood_correct, num_ood, all_ood_correct / num_ood))
        print('----------------------------------------------------------------')

        # Evaluations
        socre_save_dir = os.path.join(args.model_dir, 'epoch_{}_scores'.format(epoch))
        if not os.path.exists(socre_save_dir):
            os.makedirs(socre_save_dir)
        all_in_full_scores_file = os.path.join(socre_save_dir, 'id_misc_scores.txt')

        # eval clean acc
        test_nat_id_acc, test_nat_miscls_out_cls, _, test_nat_id_msp, test_nat_id_corr_prob, test_nat_id_out_msp, test_nat_id_out_ssp, _, _ = \
            nn_util.eval_with_out_classes(model, test_loader, NUM_IN_CLASSES, args.num_out_classes,
                                          num_v_classes=0, misc_score_file=all_in_full_scores_file)
        test_nat_id_mmsp = test_nat_id_msp.mean().item()
        test_nat_id_corr_mprob = test_nat_id_corr_prob.mean().item()
        test_nat_id_out_mmsp = test_nat_id_out_msp.mean().item()
        test_nat_id_out_mssp = test_nat_id_out_ssp.mean().item()
        print('Testing nat ID: acc: {}, miscls to out classes: {}, mean of nat-msp: {}, mean of corr nat-msp: {}, '
              'mean of nat-out-msp: {}, mean of nat-out-ssp: {}'.format(test_nat_id_acc, test_nat_miscls_out_cls,
                                                                        test_nat_id_mmsp, test_nat_id_corr_mprob,
                                                                        test_nat_id_out_mmsp, test_nat_id_out_mssp))
        # eval acc under pgd-20 attack
        test_adv_id_acc, test_adv_in_scores, test_adv_corr_scores = 0., 0., 0.
        if args.training_method in ID_CLN_MIXPGD_TRAINING_METHODS + ID_PAIR_TRAINING_METHODS and epoch > args.adv_id_warmup:
            print('----------------------------------------------------------------')
            test_adv_id_acc, test_adv_miscls_out_cls, _, test_adv_id_msp, test_adv_id_corr_prob, test_adv_id_out_msp, test_adv_id_out_ssp, _, _ \
                = pgd.eval_pgdadv_with_out_classes(model, test_loader, args.attack_test_step, args.attack_lr,
                                                   args.attack_eps, num_in_classes=NUM_IN_CLASSES,
                                                   attack_other_in=False, num_out_classes=args.num_out_classes,
                                                   num_v_classes=0, norm=args.norm, data_type='in', loss_str='ce',
                                                   best_loss=args.best_loss)
            test_adv_id_mmsp = test_adv_id_msp.mean().item()
            test_adv_id_corr_mprob = test_adv_id_corr_prob.mean().item()
            test_adv_id_out_mmsp = test_adv_id_out_msp.mean().item()
            test_adv_id_out_mssp = test_adv_id_out_ssp.mean().item()
            print('Under PGD-20 attack, testing adv id acc: {}, miscls to out_classes: {}, mean of adv-msp: {}, '
                  'mean of adv corr-prob: {}, mean of adv-out-msp: {}, mean of adv-out-ssp: {}'
                  .format(test_adv_id_acc, test_adv_miscls_out_cls, test_adv_id_mmsp, test_adv_id_corr_mprob,
                          test_adv_id_out_mmsp, test_adv_id_out_mssp))
        adv_id_auroc = 0.
        adv_id_fpr95 = 1.
        if epoch > args.schedule[0] - 1:
            # # eval detection performance under adaptive attacks
            # st = time.time()
            # ver = 'apgd-cw'
            # x_test, y_test, aa_x = eval_ood_obranch_util.attack_id(model, test_loader, NUM_IN_CLASSES, args,
            #                                                         attack_all_emps=False, best_loss=True, version=ver,
            #                                                         device=device)
            # org_incorr_indcs, succ_pertub_indcs, incorr_indcs \
            #     = eval_ood_obranch_util.check_adv_status(model, x_test, aa_x, y_test, NUM_IN_CLASSES, args.batch_size)
            # test_aa_id_acc, test_aa_id_out_cls, _, test_aa_id_msp, test_aa_id_corr_prob, _, _ , _, _ \
            #     = nn_util.eval_from_data_with_out_classes(model, aa_x, y_test, args.batch_size, NUM_IN_CLASSES,
            #                                               args.num_out_classes, num_v_classes=0)
            # test_aa_id_mmsp = test_aa_id_msp.mean().item()
            # test_aa_id_corr_mprob = test_aa_id_corr_prob.mean().item()
            # print('Under AA {}, test aa id acc: {}, misclassified to out-classes:{}, misclassified to v-classes:{}, '
            #       'mean of aa-msp: {}, mean of aa-corr-prob: {}'.format(ver, test_aa_id_acc, test_aa_id_out_cls,
            #                                                              test_aa_id_v_cls, test_aa_id_mmsp,
            #                                                              test_aa_id_corr_mprob))
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
            print('Training performance on OODs:')
            for key_odd, ood_msp in {'train_nat_ood_msp': train_nat_ood_msp,
                                     'train_adv_ood_msp': train_adv_ood_msp}.items():
                if len(ood_msp) == 0:
                    continue
                st = time.time()
                aa_ood_auroc = nn_util.auroc(test_nat_id_msp.to(storage_device), ood_msp.to(storage_device))
                aa_ood_fpr95, _ = nn_util.fpr_at_tprN(test_nat_id_msp.to(storage_device), ood_msp.to(storage_device), TPR=95)
                aa_ood_tpr95, _ = nn_util.tpr_at_tnrN(test_nat_id_msp.to(storage_device), ood_msp.to(storage_device), TNR=95)
                et = time.time()
                print('Num of {}: {}, mean of msp: {}, AUROC: {}, FPR@TPR95: {}, TPR@TNR95: {}, evaluation time: {}s'
                      .format(key_odd, ood_msp.size(0), ood_msp.mean().item(), aa_ood_auroc, aa_ood_fpr95, aa_ood_tpr95, et - st))

            print('----------------------------------------')
            print('performance on val OOD data:')
            img_size=[32, 32]
            if args.dataset == 'cifar10' or args.dataset == 'cifar100':
                out_datasets = ['places365', 'svhn', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
            elif args.dataset == 'svhn':
                out_datasets = ['cifar10', 'cifar100', 'places365', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
            elif args.dataset == 'tiny-imagenet-200':
                img_size = [64, 64]
                out_datasets = ['places365', 'dtd']
            elif args.dataset == 'tiny-imagenet-200-32x32':
                out_datasets = ['places365', 'dtd']
            id_score_dict = {'in_msp': test_nat_id_msp, 'out_ssp_minus_in_msp': test_nat_id_out_ssp - test_nat_id_msp,
                             'out_ssp': test_nat_id_out_ssp}
            ood_attack_methods = ['clean', 'adp-ce_in-out', 'apgd-adp-ce_in-out', 'apgd-adp-cw_in-out-targeted']
            for o_a_m in ood_attack_methods:
                for step in [100]:
                    st = time.time()
                    _, inputs_num, miscls_in_classes, miscls_v_classes, in_msp_dict, _, out_ssp_dict, _, v_ssp_dict \
                        = eval_ood_obranch_util.get_ood_scores(model, NUM_IN_CLASSES, args.num_out_classes,
                                                               0, socre_save_dir, out_datasets,
                                                               ood_attack_method=o_a_m, ood_batch_size=128,
                                                               attack_lr=args.attack_lr, attack_eps=args.attack_eps,
                                                               attack_all_emps=True, best_loss=True, pgd_test_step=step,
                                                               pgd_restarts=1, aa_test_step=step, aa_restarts=5,
                                                               taa_test_step=step, targets=min(NUM_IN_CLASSES, 10),
                                                               device=device, num_oods_per_set=128 * 2,
                                                               img_size=img_size, norm=args.norm)
                    v_and_out_minus_in_dict = {}
                    v_and_out_dict = {}
                    out_minus_in_dict = {}
                    for temp_key, _ in out_ssp_dict.items():
                        v_and_out_minus_in_dict[temp_key] = v_ssp_dict[temp_key] + out_ssp_dict[temp_key] - in_msp_dict[temp_key]
                        v_and_out_dict[temp_key] = v_ssp_dict[temp_key] + out_ssp_dict[temp_key]
                        out_minus_in_dict[temp_key] = out_ssp_dict[temp_key] - in_msp_dict[temp_key]
                    for sc_key, ood_values in {'in_msp': [in_msp_dict, 'in_msp'],
                                               # 'out_ssp_minus_in_msp': [out_minus_in_dict, 'r_ssp'],
                                               'out_ssp': [out_ssp_dict, 'r_ssp']}.items():
                        if sc_key not in id_score_dict:
                            continue
                        con_f = id_score_dict[sc_key]
                        conf_t = ood_values[0]
                        scoring_func = ood_values[1]
                        (_, _, _, _), (indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score), (
                        mixing_auc, mixing_fprN, mixing_tprN, mixing_score) \
                            = eval_ood_obranch_util.eval_on_signle_ood_dataset(con_f, out_datasets, conf_t, ts=[80, 85, 90, 95],
                                                                               scoring_func=scoring_func,
                                                                               storage_device=storage_device)
                        print('{}-step {} attacked OODs: with {} scoring function, indiv_auc:{}, indiv_fprN:{}, '
                              'indiv_tprN:{}, indiv_mean_score:{}'.format(step, o_a_m, sc_key, indiv_auc, indiv_fprN,
                                                                          indiv_tprN, indiv_mean_score))
                        print("{}-step {} attacked OODs: with {} scoring function, mixing_auc: {}, mixing_fprN: {}, "
                              "mixing_tprN: {}, mixing_score: {}, eval time: {}s"
                              .format(step, o_a_m, sc_key, mixing_auc, mixing_fprN, mixing_tprN, mixing_score,
                                      time.time() - st))
                        print()

        # maintain records
        training_record[epoch] = {'loss': loss, 'train_nat_id_acc': train_nat_id_acc,
                                  'train_adv_id_acc': train_adv_id_acc,
                                  'test_nat_id_acc': test_nat_id_acc, 'test_adv_id_acc': test_adv_id_acc,
                                  'test_adv_id_auc': adv_id_auroc, 'test_adv_id_fpr95': adv_id_fpr95
                                  }
        cur_cpt = {'model': model, 'optimizer': optimizer, 'record': training_record}
        if args.epochs - epoch <= args.topk_cpts:
            save_cpt(cur_cpt, args.model_dir, args.training_method, epoch)

        mom_falg = (epoch - args.delta_mom_model >= 0 and (epoch - args.delta_mom_model == mom_model[0]))
        if mom_falg or ((not args.o_multi_stage) and (not args.oadv_multi_stage)): # or epoch <= args.delta_mom_model:
            # if epoch - args.delta_mom_model == mom_model[0] or epoch == 1 or args.delta_mom_model == 0:
            mom_model[1], mom_model[0] = copy.deepcopy(model), epoch
            print('Info, setted model of epoch: {} as mom_model >>>>>'.format(mom_model[0]))

    return training_record


def get_model(model_name, num_in_classes=10, num_out_classes=0, num_v_classes=0, normalizer=None, dataset='cifar10'):
    size_3x32x32 = ['svhn', 'cifar10', 'cifar100', 'tiny-imagenet-200-32x32']
    size_3x64x64 = ['tiny-imagenet-200']
    size_3x224x224 = ['imagenet']
    if dataset in size_3x32x32:
        if model_name == 'wrn-34-10':
            return wideresnet.WideResNet(depth=34, widen_factor=10, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-28-10':
            return wideresnet.WideResNet(depth=28, widen_factor=10, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-40-4':
            return wideresnet.WideResNet(depth=40, widen_factor=4, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-40-2':
            return wideresnet.WideResNet(depth=40, widen_factor=2, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-18':
            return resnet.ResNet18(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-34':
            return resnet.ResNet34(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-50':
            return resnet.ResNet50(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'densenet':
            return densenet.DenseNet3(100, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        else:
            raise ValueError('un-supported model: {0}', model_name)
    elif dataset in size_3x64x64:
        if model_name == 'wrn-34-10':
            return wideresnet.WideResNet(depth=34, widen_factor=10, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-28-10':
            return wideresnet.WideResNet(depth=28, widen_factor=10, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-40-4':
            return wideresnet.WideResNet(depth=40, widen_factor=4, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-40-2':
            return wideresnet.WideResNet(depth=40, widen_factor=2, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-18':
            return resnet_64x64.resnet18(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-34':
            return resnet_64x64.resnet34(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-50':
            return resnet_64x64.resnet50(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnext-50':
            return resnext_64x64.resnext50_32x4d(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
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
    global AUX_OOD_CLASSES

    # setup data loader
    normalizer = None
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    transform_test = T.Compose([T.ToTensor()])
    aux_id_train_loader = None
    aux_id_transform_train = T.Compose([
        T.ToTensor(),
        T.ToPILImage(),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    if args.dataset == 'cifar10':
        # normalizer = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        dataloader = torchvision.datasets.CIFAR10
        train_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar10', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar10', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
        if args.aux_id_file != '':
            aux_id_train_loader = torch.utils.data.DataLoader(
                AuxInDistributionImages(aux_file=args.aux_id_file, transform=aux_id_transform_train),
                batch_size=args.aux_id_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        # normalizer = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        dataloader = torchvision.datasets.CIFAR100
        train_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar100', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar100', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
        if args.aux_id_file != '':
            aux_id_train_loader = torch.utils.data.DataLoader(
                AuxInDistributionImages(aux_file=args.aux_id_file, transform=aux_id_transform_train),
                batch_size=args.aux_id_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'svhn':
        # normalizer = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform_train = T.Compose([
            T.ToTensor(),
        ])
        transform_test = T.Compose([
            T.ToTensor(),
        ])
        svhn_train = torchvision.datasets.SVHN(root='../datasets/svhn/', download=True, transform=transform_train,
                                               split='train')
        svhn_test = torchvision.datasets.SVHN(root='../datasets/svhn/', download=True, transform=transform_test,
                                              split='test')
        train_loader = torch.utils.data.DataLoader(dataset=svhn_train, batch_size=args.batch_size, shuffle=True,
                                                   **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset=svhn_test, batch_size=args.batch_size, shuffle=False,
                                                  **kwargs)
        if args.aux_id_file != '':
            aux_id_train_loader = torch.utils.data.DataLoader(
                AuxInDistributionImages(aux_file=args.aux_id_file, transform=aux_id_transform_train),
                batch_size=args.aux_id_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'tiny-imagenet-200-32x32':
        data_dir = '../datasets/tiny-imagenet-200/'
        # normalizer = T.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821])
        transform_train = T.Compose([
            T.RandomResizedCrop(32),
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        transform_test = T.Compose([
            T.Resize(32),
            T.ToTensor(),
        ])
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'tiny-imagenet-200':
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        data_dir = '../datasets/tiny-imagenet-200/'
        # normalizer = T.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821])
        transform_train = T.Compose([
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        transform_test = T.Compose([
            T.ToTensor(),
        ])
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
        if args.aux_id_file != '':
            aux_id_transform_train = T.Compose([
                T.ToTensor(),
                T.ToPILImage(),
                T.RandomCrop(64, padding=8),
                T.RandomHorizontalFlip(),
                T.ToTensor()
            ])
            aux_id_train_loader = torch.utils.data.DataLoader(
                AuxTiny200Images(aux_path=args.aux_id_file, file_x='tiny_edm_1m_x.npy', file_y='tiny_edm_1m_y.npy',
                                 length=1000000, img_size=64, transform=aux_id_transform_train),
                batch_size=args.aux_id_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError('un-supported dataset: {0}'.format(args.dataset))

    ood_mining_batch_size = args.ood_batch_size
    if args.mine_ood:
        ood_mining_batch_size = max(args.ood_batch_size, 400)
    num_fixed_ood = args.num_fixed_ood
    if args.fix_ood and args.num_fixed_ood == -1:
        num_fixed_ood = int(len(train_loader) * (float(ood_mining_batch_size) / float(args.batch_size)) * args.batch_size)
        print('num_fixed_ood:', num_fixed_ood)

    out_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.auxiliary_dataset == '80m_tiny_images':
        ood_transform_train = T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        ood_train_loader = torch.utils.data.DataLoader(
            TinyImages(transform=ood_transform_train, num_fixed_x=num_fixed_ood, tiny_file=args.ood_file),
            batch_size=ood_mining_batch_size, shuffle=False, **out_kwargs)
    elif args.auxiliary_dataset == '300k-random-images':
        ood_transform_train = T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        ood_train_loader = torch.utils.data.DataLoader(
            RandomImages(transform=ood_transform_train, tiny_file=args.ood_file), batch_size=ood_mining_batch_size,
            shuffle=False, **out_kwargs)
    elif args.auxiliary_dataset == 'downsampled-imagenet':
        ood_mining_batch_size = args.ood_batch_size
        ood_transform_train = T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        img_loader = ImageNetEXCIFAR(transform=ood_transform_train, id_type=args.dataset, exclude_cifar=True,
                                     excl_simi=0.35, img_size=32, imagenet_dir=args.ood_file,
                                     num_included_classes=args.num_included_ood_classes, num_fixed_x=num_fixed_ood)
        AUX_OOD_CLASSES = img_loader.get_selected_classes()
        for indx in range(0, len(AUX_OOD_CLASSES)):
            ood_class = AUX_OOD_CLASSES[indx]
            OOD_CLS_2_NEW_CLS[ood_class] = NUM_IN_CLASSES + indx % args.num_out_classes
        print('OOD_CLS_2_NEW_CLS:', OOD_CLS_2_NEW_CLS)
        if args.ood_batch_size == -1:
            ood_mining_batch_size = int(len(img_loader) / len(train_loader))
        else:
            if args.mine_ood:
                ood_mining_batch_size = max(args.ood_batch_size, 400)
        print('Info, ood_batch_size is (re-)set to {}'.format(ood_mining_batch_size))
        ood_train_loader = torch.utils.data.DataLoader(img_loader, batch_size=ood_mining_batch_size, shuffle=False, **out_kwargs)
    elif args.auxiliary_dataset == 'tired-imagenet':
        ood_transform_train = T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.Resize(32),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        img_loader = TieredImageNet(transform=ood_transform_train, id_type=args.dataset, exclude_cifar=True,
                                    data_dir=args.ood_file, num_included_gene_classes=args.num_included_ood_classes)
        AUX_OOD_CLASSES = img_loader.get_selected_classes()
        for indx in range(0, len(AUX_OOD_CLASSES)):
            ood_class = AUX_OOD_CLASSES[indx]
            OOD_CLS_2_NEW_CLS[ood_class] = NUM_IN_CLASSES + indx % args.num_out_classes
        print('OOD_CLS_2_NEW_CLS:', OOD_CLS_2_NEW_CLS)
        if args.mine_ood:
            ood_mining_batch_size = max(ood_mining_batch_size, 400)
        if ood_mining_batch_size != args.ood_batch_size:
            print('INFO, ood_size is automatically set to {}'.format(ood_mining_batch_size))
        ood_train_loader = torch.utils.data.DataLoader(img_loader, batch_size=ood_mining_batch_size, shuffle=False, **out_kwargs)

    elif args.auxiliary_dataset == 'downsampled-imagenet-800':
        ood_mining_batch_size = args.ood_batch_size
        out_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        if '32x32' in args.dataset:
            ood_transform_train = T.Compose([
                T.ToTensor(),
                T.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
            img_size = 32
        else:
            ood_transform_train = T.Compose([
                T.ToTensor(),
                T.ToPILImage(),
                T.RandomCrop(64, padding=8),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
            img_size = 64
        img_loader = ImageNet800(transform=ood_transform_train, imagenet_dir=args.ood_file, img_size=img_size,
                                 num_included_classes=args.num_included_ood_classes, num_fixed_x=num_fixed_ood,)
        AUX_OOD_CLASSES = img_loader.get_selected_classes()
        for indx in range(0, len(AUX_OOD_CLASSES)):
            ood_class = AUX_OOD_CLASSES[indx]
            OOD_CLS_2_NEW_CLS[ood_class] = NUM_IN_CLASSES + indx % args.num_out_classes
        print('OOD_CLS_2_NEW_CLS:', OOD_CLS_2_NEW_CLS)
        if args.ood_batch_size == -1:
            ood_mining_batch_size = int(len(img_loader) / len(train_loader))
        else:
            if args.mine_ood:
                ood_mining_batch_size = max(args.ood_batch_size, 400)
        if ood_mining_batch_size != args.ood_batch_size:
            print('INFO, ood_size is automatically set to {}'.format(ood_mining_batch_size))
        ood_train_loader = torch.utils.data.DataLoader(img_loader, batch_size=ood_mining_batch_size, shuffle=False, **out_kwargs)
    else:
        ood_train_loader = None
    cudnn.benchmark = True

    # init model, Net() can be also used here for training
    model = get_model(args.model_name, num_in_classes=NUM_IN_CLASSES, num_out_classes=args.num_out_classes,
                      num_v_classes=0, normalizer=normalizer, dataset=args.dataset).to(device)
    if len(args.gpus) > 1:
        model = nn.DataParallel(model.to(device), device_ids=args.gpus, output_device=args.gpus[0])
    else:
        model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print('===================================================================================================')
    print('args:', args)
    print('===================================================================================================')

    train(model, train_loader, optimizer, test_loader, aux_id_train_loader, ood_train_loader, normalizer)


if __name__ == '__main__':
    main()

