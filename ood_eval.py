from __future__ import print_function
import os
import torch
import argparse
import torchvision.transforms as T
from utils import nn_util, eval_ood_util
from models import wideresnet, resnet, densenet
import torchvision
import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR OOD Detection Evaluation')
parser.add_argument('--model_name', default='wrn-40-4',
                    help='model name, wrn-28-10, wrn-34-10, wrn-40-4, resnet-18, resnet-50')
parser.add_argument('--num_v_classes', default=0, type=int,
                    help='the number of label-irrelevant nodes in last layer')
parser.add_argument('--alpha', default=0., type=float, help='total confidence of virtual dayu classes')
parser.add_argument('--dataset', default='cifar10', help='dataset: svhn, cifar10 or cifar100')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--attack_eps', default=8.0, type=float, help='perturbation')
parser.add_argument('--attack_lr', default=2.0, type=float, help='perturb step size')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpuid', type=int, default=0, help='The ID of GPU.')
parser.add_argument('--pgd_test_step', default=20, type=int, help='perturb number of steps in PGD')
parser.add_argument('--aa_test_step', default=100, type=int, help='perturb number of steps in Auto-PGD')
parser.add_argument('--model_file', default='./dnn_models/dataset/model.pt', help='file path of src model')

parser.add_argument('--save_socre_dir', default='', help='dir for saving scores')
parser.add_argument('--ood_batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--correct_scores', action='store_true', default=False,
                    help='whether use correct-scores instead of real-scores to calculate AUROC and FPR')
parser.add_argument('--storage_device', default='cpu', help='device for computing auroc and fpr: cuda or cpu')

parser.add_argument('--model_dir', default='', help='directory of model for saving checkpoint')
parser.add_argument('--training_method', default='clean',
                    help='training method: clean, clean-pgd-ce, pair-pgd-ce, pgd-ce')
parser.add_argument('--st_epoch', default=150, type=int, help='start epoch')
parser.add_argument('--end_epoch', default=201, type=int, help='end epoch')
parser.add_argument('--targets', default=9, type=int, help='number of targets in multi-target attack')
parser.add_argument('--taa_test_step', default=100, type=int, help='perturb number of steps in Targeted Auto-PGD')

args = parser.parse_args()

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

if args.dataset == 'cifar10':
    NUM_IN_CLASSES = 10
elif args.dataset == 'svhn':
    NUM_IN_CLASSES = 10
elif args.dataset == 'cifar100':
    NUM_IN_CLASSES = 100
else:
    raise ValueError('error dataset: {0}'.format(args.dataset))

if args.attack_lr > 1:
    args.attack_lr = args.attack_lr / 255
if args.attack_eps > 1:
    args.attack_eps = args.attack_eps / 255
if args.targets < 0:
    args.targets = NUM_IN_CLASSES
    print('INFO, args.targets is not given, I set it to {}'.format(NUM_IN_CLASSES - 1))

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
storage_device = torch.device(args.storage_device)

if args.save_socre_dir == '' and args.model_file != '':
    args.save_socre_dir = args.model_file + '.eval-scores'
    if not os.path.exists(args.save_socre_dir):
        os.makedirs(args.save_socre_dir)
    print('Warning, args.save_socre_dir is not given, I set it to: {}'.format(args.save_socre_dir))

def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


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


def eval_main():
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform_test = T.Compose([T.ToTensor()])
    if args.dataset == 'cifar10':
        normalizer = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        dataloader = torchvision.datasets.CIFAR10
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar10/', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)

    elif args.dataset == 'cifar100':
        normalizer = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        dataloader = torchvision.datasets.CIFAR100
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar100/', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'svhn':
        normalizer = None
        svhn_test = torchvision.datasets.SVHN(root='../datasets/svhn/', download=True, transform=T.ToTensor(),
                                              split='test')
        test_loader = torch.utils.data.DataLoader(dataset=svhn_test, batch_size=args.batch_size, shuffle=False,
                                                  **kwargs)
    # elif args.dataset == 'mnist':
    #     # normalizer T.Normalize((0.1307,), (0.3081,))
    #     normalizer = None
    #     test_trans = T.Compose([
    #         T.ToTensor(),
    #     ])
    #     mnist_test = torchvision.datasets.MNIST('../datasets/mnist', train=False, download=True, transform=test_trans)
    #     test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=args.batch_size, shuffle=False,
    #                                               **kwargs)
    # elif args.dataset == 'fashion-mnist':
    #     normalizer = None
    #     test_trans = T.Compose([
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.286], std=[0.352])
    #     ])
    #     fashion_mnist_test = torchvision.datasets.FashionMNIST(root='../datasets/fashion-mnist', train=False,
    #                                                            download=True, transform=test_trans)
    #     test_loader = torch.utils.data.DataLoader(fashion_mnist_test, batch_size=args.batch_size, shuffle=False)
    else:
        raise ValueError('un-supported dataset: {0}'.format(args.dataset))

    print('args:', args)
    print('================================================================')

    model = get_model(args.model_name, num_in_classes=NUM_IN_CLASSES, num_out_classes=0, num_v_classes=args.num_v_classes,
                      normalizer=normalizer).to(device)
    if args.model_dir == '':
        checkpoint = torch.load(args.model_file)
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint)
        model = model.to(device)
        print('Successfully loaded model from:{}'.format(args.model_file))

        eval_ood_util.eval_in_and_out(model, test_loader, NUM_IN_CLASSES, args, device=device,
                                      storage_device=storage_device)
    else:
        pick_cpts(model, test_loader, args)


def pick_cpts(model, test_loader, args):
    st_epoch = args.st_epoch
    end_epoch = args.end_epoch
    for epoch in range(st_epoch, end_epoch):
        model_file = os.path.join(args.model_dir, '{0}_model_epoch{1}.pt'.format(args.training_method, epoch))
        if not os.path.exists(model_file):
            print('cpt file is not found: {}'.format(model_file))
            continue
        else:
            print('===============================================================================================')
            print('eval cpt epoch {} from {}'.format(epoch, model_file))
        model.load_state_dict(torch.load(model_file))
        model = model.to(device)
        socre_save_dir = os.path.join(args.model_dir, 'epoch_{}_scores'.format(epoch))
        if not os.path.exists(socre_save_dir):
            os.makedirs(socre_save_dir)
        all_id_misc_scores_file = os.path.join(socre_save_dir, 'id_misc_scores.txt')
        all_id_logits_file = os.path.join(socre_save_dir, 'id_logits.npy')

        # eval clean acc
        nat_id_acc, nat_id_v_cls, nat_id_v_cls_corr, nat_id_msp, nat_id_corr_prob, nat_id_v_msp, nat_id_v_ssp = \
            nn_util.eval(model, test_loader, NUM_IN_CLASSES, num_v_classes=args.num_v_classes,
                         misc_score_file=all_id_misc_scores_file, lotis_file=all_id_logits_file)
        nat_id_mmsp = nat_id_msp.mean().item()
        nat_id_corr_mprob = nat_id_corr_prob.mean().item()
        nat_id_vssp_minus_inmsp = nat_id_v_ssp - nat_id_msp
        nat_id_m_vssp_minus_inmsp = nat_id_vssp_minus_inmsp.mean().item()
        print('Tetsing nat ID acc: {}, nat miscls v-classes: {}, nat micls v-classes corr: {}, mean of in-msp: {}, '
              'mean of corr-prob: {}, mean of (v_ssp - in-msp)'.format(nat_id_acc, nat_id_v_cls, nat_id_v_cls_corr,
                                             nat_id_mmsp, nat_id_corr_mprob, nat_id_m_vssp_minus_inmsp))

        '''
        print('------------------------------------------------------------------------------------------------------')
        print('performance on adv ID data:')
        Ns = [80, 85, 90, 95]
        if args.num_v_classes>0:
            id_attack_methods = ['ce', 'cw',
                                 'adp-ce_in', 'adp-ce_in-v', 'adp-cw_in', 'adp-cw_in-v', 'apgd-cw',
                                 'apgd-adp-ce_in', 'apgd-ce_in-v', 'apgd-adp-cw_in', 'apgd-adp-cw_in-v',
                                 'apgd-adp-ce_in-targeted', 'apgd-adp-ce_in-v-targeted',
                                 'apgd-adp-cw_in-targeted', 'apgd-adp-cw_in-v-targeted', 'apgd-adp-cw-targeted',
                                 ]
        else:
            id_attack_methods = ['ce', 'cw', 'adp-ce_in', 'adp-cw_in', 'apgd-adp-ce_in', 'apgd-adp-cw_in', 'apgd-cw',
                                 'apgd-adp-ce_in-targeted', 'apgd-adp-cw_in-targeted', 'apgd-adp-cw-targeted',
                                ]
        for i_a_m in id_attack_methods:
            # for best_loss in best_losses:
            print('---------------------------------------------------------------------------------------')
            st = time.time()
            x_test, y_test, adv_x = eval_ood_util.attack_id(model, test_loader, NUM_IN_CLASSES, args, version=i_a_m,
                                                            attack_all_emps=True, best_loss=True,
                                                            num_ids=args.batch_size * 4, device=device)
            org_incorr_indcs, succ_pertub_indcs, incorr_indcs = \
                eval_ood_util.check_adv_status(model, x_test, adv_x, y_test, NUM_IN_CLASSES, args.batch_size)
            org_corr_adv_indcs = ~org_incorr_indcs
            in_score_file = os.path.join(args.save_socre_dir, i_a_m + '_test_id_scores.txt')
            adv_id_acc, adv_id_v_cls, adv_id_v_cls_corr, adv_id_msp, adv_id_corr_prob, adv_id_v_msp, adv_id_v_ssp \
                = nn_util.eval_from_data(model, adv_x, y_test, args.batch_size, NUM_IN_CLASSES, args.num_v_classes,
                                         misc_score_file=in_score_file)
            adv_id_corr_mprob = adv_id_corr_prob.mean().item()
            adv_id_mmsp = adv_id_msp.mean().item()
            adv_id_vssp_minus_inmsp = adv_id_v_ssp - adv_id_msp
            adv_id_m_vssp_minus_inmsp = adv_id_vssp_minus_inmsp.mean().item()
            print('Under {}, test adv id acc: {}, misclassified to v-classes: {}, misclassified to v-classes corr: {}, '
                  'mean of in-msp: {}, mean of corr-prob: {}, mean of (v_ssp - in-msp): {}'
                  .format(i_a_m, adv_id_acc, adv_id_v_cls, adv_id_v_cls_corr, adv_id_mmsp, adv_id_corr_mprob,
                          adv_id_m_vssp_minus_inmsp))
            if args.num_v_classes > 0:
                sc_kvs = {'in-msp': [nat_id_msp, adv_id_msp, 'in_msp'],
                          'v-ssp_minus_in-msp': [nat_id_vssp_minus_inmsp, adv_id_vssp_minus_inmsp, 'r_ssp']}
            else:
                sc_kvs = {'in-msp': [nat_id_msp, adv_id_msp, 'in_msp']}
            for key_ind, ind in {'org_corr_adv_indcs': org_corr_adv_indcs,
                                 'succ_pertub_indcs': succ_pertub_indcs}.items():
                for sc_key, sc_vals in sc_kvs.items():
                    conf_f = sc_vals[0]
                    conf_t = sc_vals[1]
                    sc_func = sc_vals[2]
                    mean_adv_score = conf_t[ind].mean().item()
                    adv_auroc = nn_util.auroc(conf_f, conf_t[ind], scoring_func=sc_func)
                    adv_fprNs = {}
                    adv_tprNs = {}
                    for N in Ns:
                        adv_fprN, _ = nn_util.fpr_at_tprN(conf_f, conf_t, TPR=N, scoring_func=sc_func)
                        adv_tprN, _ = nn_util.tpr_at_tnrN(conf_f, conf_t, TNR=N, scoring_func=sc_func)
                        adv_fprNs[N] = adv_fprN
                        adv_tprNs[N] = adv_tprN
                    print('with scoring function: {}, num of {}: {}, mean of in-msp: {}, AUROC: {}, FPR@TPR-Ns: {}, '
                          'TPR@TNR-Ns: {}, eval time: {}s'.format(sc_key, key_ind, ind.sum().item(), mean_adv_score,
                                                                  adv_auroc, adv_fprNs, adv_tprNs, time.time() - st))
                print()
        '''

        '''
        print('------------------------------------------------------------------------------------------------------')
        print('Robustness of the classifier under Auto-Attack')
        x_test, y_test, adv_x = eval_ood_util.attack_id(model, test_loader, NUM_IN_CLASSES, args, version='standard-aa',
                                                        attack_all_emps=False, best_loss=False, device=device)
        adv_id_acc, adv_id_v_cls, adv_id_v_cls_corr, adv_id_msp, adv_id_corr_prob, adv_id_v_msp, adv_id_v_ssp \
            = nn_util.eval_from_data(model, adv_x, y_test, args.batch_size, NUM_IN_CLASSES, args.num_v_classes)
        adv_id_corr_mprob = adv_id_corr_prob.mean().item()
        adv_id_mmsp = adv_id_msp.mean().item()
        print('Under Auto-Attack, test adv id acc: {}, misclassified to v-classes: {}, misclassified to v-classes corr: {}, '
              'mean of in-msp: {}, mean of corr-prob: {}'
              .format(adv_id_acc, adv_id_v_cls, adv_id_v_cls_corr, adv_id_mmsp, adv_id_corr_mprob))
        '''

        print('------------------------------------------------------------------------------------------------------')
        print('performance on val OOD data:')
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            out_datasets = ['places365', 'svhn', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
        elif args.dataset == 'svhn':
            out_datasets = ['cifar10', 'cifar100', 'places365', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']

        id_score_dict = {'in_msp': nat_id_msp}
        ood_attack_method_arrs = [['clean'],
                                  ['adp-ce_in'], ['adp-cw_in'],
                                  ['apgd-adp-ce_in'], ['apgd-adp-cw_in'],
                                  ['apgd-adp-ce_in-targeted'], ['apgd-adp-cw_in-targeted'],
                                  ['clean', 'adp-ce_in', 'apgd-adp-cw_in-targeted'],
                                  ]

        for o_a_ms in ood_attack_method_arrs:
            print('-----------------------------------------------')
            st = time.time()
            _, inputs_num, _, ood_in_msp_dict, _, ood_v_msp_dict, ood_v_ssp_dict \
                = eval_ood_util.get_ood_scores(model, NUM_IN_CLASSES, socre_save_dir, out_datasets, args,
                                               ood_attack_methods=o_a_ms, ood_batch_size=args.ood_batch_size,
                                               attack_all_emps=True, best_loss=True, pgd_test_step=args.pgd_test_step,
                                               aa_test_step=args.aa_test_step, taa_test_step=args.taa_test_step,
                                               targets=args.targets, device=device,
                                               num_oods_per_set=args.ood_batch_size * 4)
            ood_vssp_inmsp_dict = {}
            ood_vmsp_inmsp_dict = {}
            for temp_key, _ in ood_v_ssp_dict.items():
                ood_vssp_inmsp_dict[temp_key] = ood_v_ssp_dict[temp_key] - ood_in_msp_dict[temp_key]
                ood_vmsp_inmsp_dict[temp_key] = ood_v_msp_dict[temp_key] - ood_in_msp_dict[temp_key]

            for sc_key, ood_values in {'in_msp': [ood_in_msp_dict, 'in_msp'],
                                       'v_msp': [ood_v_msp_dict, 'r_ssp'],
                                       'v_ssp': [ood_v_ssp_dict, 'r_ssp'],
                                       'v_msp-in_msp': [ood_vmsp_inmsp_dict, 'r_ssp'],
                                       'v_ssp-in_msp': [ood_vssp_inmsp_dict, 'r_ssp'],
                                       }.items():
                if sc_key not in id_score_dict:
                    continue
                con_f = id_score_dict[sc_key]
                print('=====> using {} as scoring function =====>'.format(sc_key))
                conf_t = ood_values[0]
                scoring_func = ood_values[1]
                (_, _, _, _), (indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score), (mixing_auc, mixing_fprN, mixing_tprN, mixing_score) \
                    = eval_ood_util.eval_on_signle_ood_dataset(con_f, out_datasets, conf_t, ts=[80, 85, 90, 95],
                                                               scoring_func=scoring_func, storage_device=storage_device)
                print("Under attack: {}".format(o_a_ms))
                print("Performance on Individual OOD set: indiv_auc: {}, indiv_fprN: {}, indiv_tprN: {}, indiv_mean_score: {}, eval time: {}s"
                      .format(indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score, time.time() - st))
                print("Performance on Mixed OOD set: mixing_auc: {}, mixing_fprN: {}, mixing_tprN: {}, mixing_score: {}, eval time: {}s"
                      .format(mixing_auc, mixing_fprN, mixing_tprN, mixing_score, time.time() - st))
            print()


def test_transfer():
    from attacks import pgd

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform_test = T.Compose([T.ToTensor()])
    if args.dataset == 'cifar10':
        # normalizer = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        dataloader = torchvision.datasets.CIFAR10
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar10/', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        # normalizer = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        dataloader = torchvision.datasets.CIFAR100
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar100/', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'svhn':
        # normalizer = None
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        svhn_test = torchvision.datasets.SVHN(root='../datasets/svhn/', download=True, transform=transform_test,
                                              split='test')
        test_loader = torch.utils.data.DataLoader(dataset=svhn_test, batch_size=args.batch_size, shuffle=False,
                                                  **kwargs)
    else:
        raise ValueError('un-supported dataset: {0}'.format(args.dataset))
    print('================================================================')

    src_model = get_model(args.model_name, num_in_classes=NUM_IN_CLASSES, num_out_classes=0,
                          num_v_classes=args.num_v_classes, normalizer=None).to(device)
    checkpoint = torch.load(args.model_file, map_location='cpu')
    # model.load_state_dict(checkpoint['state_dict'])
    src_model.load_state_dict(checkpoint)
    src_model = src_model.to(device)
    print('Successfully loaded model from:{}'.format(args.model_file))

    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_adv_x = pgd.fgsm_attack(src_model, batch_x, batch_y, random_init=False, attack_eps=args.attack_eps,
                                      num_in_classes=NUM_IN_CLASSES, num_out_classes=0)

        # batch_adv_x = pgd.pgd_attack(src_model, batch_x, batch_y, attack_step=10,
        #                              attack_lr=args.attack_lr, attack_eps=args.attack_eps,
        #                              num_in_classes=NUM_IN_CLASSES, num_out_classes=0)
        with torch.no_grad():
            nat_logits = src_model(batch_x)
            nat_pred = torch.max(nat_logits, dim=1)[1]
            nat_acc_indcs = nat_pred == batch_y
            adv_logits = src_model(batch_adv_x)
            adv_pred = torch.max(adv_logits, dim=1)[1]
            adv_acc_indcs = adv_pred == batch_y
            print('acc:{}, rob acc:{}'.format(nat_acc_indcs.sum().item() / len(nat_acc_indcs),
                                              adv_acc_indcs.sum().item() / len(adv_acc_indcs)))

            allowed_indcs = torch.logical_and(nat_acc_indcs, ~adv_acc_indcs)
            allowed_nat_x = batch_x[allowed_indcs]
            allowed_adv_x = batch_adv_x[allowed_indcs]
            allowed_y = batch_y[allowed_indcs]
            print("number of allowed_nat_x: {}".format(len(allowed_nat_x)))
            avd_succ_perpertub_ratio=0
            for i in range(0, len(allowed_adv_x)):
                trans_pertub = allowed_adv_x[i:i+1] - allowed_nat_x[i:i+1]
                trans_x = []
                trans_y = []
                for j in range(0, len(allowed_adv_x)):
                    if i != j:
                        trans_x.append(allowed_nat_x[j:j + 1] + trans_pertub)
                        trans_y.append(allowed_y[j])
                trans_x = torch.cat(trans_x, dim=0)
                trans_y = torch.tensor(trans_y).to(device)
                # print('trans_x.size:', trans_x.size(), 'trans_y.size:', trans_y.size())
                trans_adv_logits = src_model(trans_x)
                trans_pred = torch.max(trans_adv_logits, dim=1)[1]
                tranc_acc_indcs = trans_pred == trans_y
                succ_perpertub = len(tranc_acc_indcs) - tranc_acc_indcs.sum().item()
                print('i:{}, src y: {}, succ_perpertub ratio: {} ({}/{})'.format(i, allowed_y[i],
                                                                                 succ_perpertub / len(tranc_acc_indcs),
                                                                                 succ_perpertub, len(tranc_acc_indcs)))
                avd_succ_perpertub_ratio += succ_perpertub / len(tranc_acc_indcs)
            avd_succ_perpertub_ratio = avd_succ_perpertub_ratio / len(allowed_adv_x)
            print('avd_succ_perpertub_ratio:', avd_succ_perpertub_ratio)
        break


if __name__ == '__main__':
    # test_transfer()
    eval_main()
