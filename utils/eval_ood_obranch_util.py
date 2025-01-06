import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import utils.svhn_loader as svhn
import numpy as np

from attacks import pgd
from utils import nn_util


def get_ood_abs_dir():
    parent = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(parent)
    parent = os.path.dirname(parent)
    ood_abs_dir = parent + "/datasets/ood_datasets/"
    return ood_abs_dir


def pick_worst_images(model, num_in_classes, images, adv_images):
    with torch.no_grad():
        model.eval()
        inputs = images.detach().clone()
        nat_outputs = F.softmax(model(images), dim=1)
        # nat_max_in_scores, _ = torch.max(nat_outputs[:, :num_in_classes], dim=1)
        nat_ood_scores = torch.sum(nat_outputs[:, num_in_classes:], dim=1)

        adv_outputs = F.softmax(model(adv_images), dim=1)
        # adv_max_in_scores, _ = torch.max(adv_outputs[:, :num_in_classes], dim=1)
        adv_ood_scores = torch.sum(adv_outputs[:, num_in_classes:], dim=1)

        indices = adv_ood_scores < nat_ood_scores
        inputs[indices] = adv_images[indices]
    return inputs


def get_ood_scores(model, num_in_classes, base_dir, out_datasets, args, ood_attack_methods=['clean'], ood_batch_size=256,
                   attack_other_in=False, attack_all_emps=False, best_loss=True, pgd_test_step=20, aa_test_step=100,
                   taa_test_step=100, targets=9, num_oods_per_set=float('inf'), device=torch.device("cuda")):
    from autoattack import AutoAttack
    num_out_classes = 0
    if 'num_out_classes' in args:
        num_out_classes = args.num_out_classes
    num_v_classes = 0
    if 'num_v_classes' in args:
        num_v_classes = args.num_v_classes

    saved_logits_files = []
    all_in_msps = {}
    all_out_msps = {}
    all_out_ssps = {}
    all_v_msps = {}
    all_v_ssps = {}
    all_miscls_in_classes = {}
    all_miscls_v_classes = {}
    all_inputs_num = {}

    ood_abs_dir = get_ood_abs_dir()

    for out_dataset in out_datasets:
        out_save_dir = os.path.join(base_dir, out_dataset)
        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        if out_dataset == 'svhn':
            testset_out = svhn.SVHN(ood_abs_dir + '/svhn/', split='test', transform=T.ToTensor(), download=True)
            test_out_loader = torch.utils.data.DataLoader(testset_out, batch_size=ood_batch_size, shuffle=False,
                                                          num_workers=1)
        elif out_dataset == 'cifar10':
            dataloader = torchvision.datasets.CIFAR10
            test_out_loader = torch.utils.data.DataLoader(
                dataloader(ood_abs_dir + '/cifar10/', train=False, download=True, transform=T.ToTensor()),
                batch_size=ood_batch_size, shuffle=False, num_workers=1)
        elif out_dataset == 'cifar100':
            dataloader = torchvision.datasets.CIFAR100
            test_out_loader = torch.utils.data.DataLoader(
                dataloader(ood_abs_dir + '/cifar100/', train=False, download=True, transform=T.ToTensor()),
                batch_size=ood_batch_size, shuffle=False, num_workers=1)
        elif out_dataset == 'dtd':
            testset_out = torchvision.datasets.ImageFolder(root=ood_abs_dir + "/dtd/images", transform=T.Compose(
                [T.Resize(32), T.CenterCrop(32), T.ToTensor()]))
            test_out_loader = torch.utils.data.DataLoader(testset_out, batch_size=ood_batch_size, shuffle=False,
                                                          num_workers=1)
        elif out_dataset == 'places365':
            testset_out = torchvision.datasets.ImageFolder(root=ood_abs_dir + "/places365/test_subset",
                                                           transform=T.Compose(
                                                               [T.Resize(32), T.CenterCrop(32), T.ToTensor()]))
            test_out_loader = torch.utils.data.DataLoader(testset_out, batch_size=ood_batch_size, shuffle=False,
                                                          num_workers=1)
        else:
            testset_out = torchvision.datasets.ImageFolder(ood_abs_dir + "/{}".format(out_dataset), transform=T.Compose(
                [T.Resize(32), T.CenterCrop(32), T.ToTensor()]))
            test_out_loader = torch.utils.data.DataLoader(testset_out, batch_size=ood_batch_size, shuffle=False,
                                                          num_workers=1)

        st = time.time()
        saved_logits_file = os.path.join(out_save_dir, "{}_test_ood_logits.npy".format(ood_attack_methods))

        saved_misc_score_file = os.path.join(out_save_dir, "{}_test_ood_misc_scores.txt".format(ood_attack_methods))
        f_misc_scores = open(saved_misc_score_file, 'w')

        # print("Processing out-of-distribution images")
        N = len(test_out_loader.dataset)
        count = 0
        cur_in_msps = []
        cur_out_msps = []
        cur_out_ssps = []
        cur_v_msps = []
        cur_v_ssps = []
        miscls_in_classes = 0
        miscls_v_classes = 0
        cur_data_logits = []
        for j, data in enumerate(test_out_loader):
            if (j + 1) * ood_batch_size > num_oods_per_set:
                break
            # print('evaluating detection performance on {} {}/{}'.format(out_dataset, j+1, len(test_out_loader)))
            images, labels = data
            images = images.to(device)
            curr_batch_size = images.shape[0]

            adv_inputs = images + 0.
            for ood_attack_method in ood_attack_methods:
                # print("INFO, Using '{}' to eval OOD performance ({}) on batch {}/{}"
                #       .format(ood_attack_method, out_dataset, j, len(test_out_loader)))
                if ood_attack_method in ['adp-ce_in', 'adp-ce_out', 'adp-ce_v', 'adp-ce_in-out', 'adp-ce_in-v','adp-ce_in-out-v',
                                         'cw', 'adp-cw_in', 'adp-cw_out', 'adp-cw_v', 'adp-cw_in-out', 'adp-cw_in-v', 'adp-cw_in-out-v']:
                    loss_str = ood_attack_method
                    pgd_inputs = pgd.pgd_attack(model, images, None, attack_step=pgd_test_step,
                                            attack_lr=args.attack_lr, attack_eps=args.attack_eps, loss_str=loss_str,
                                            num_in_classes=num_in_classes, attack_other_in=attack_other_in,
                                            num_out_classes=num_out_classes, num_v_classes=num_v_classes, data_type='out',
                                            best_loss=best_loss)
                    adv_inputs = pick_worst_images(model, num_in_classes, adv_inputs, pgd_inputs)
                elif ood_attack_method in ['apgd-ce', 'apgd-adp-ce_in', 'apgd-adp-ce_out', 'apgd-adp-ce_v',
                                           'apgd-adp-ce_in-v', 'apgd-adp-ce_in-out', 'apgd-adp-ce_in-out-v',
                                           'apgd-cw', 'apgd-adp-cw_in', 'apgd-adp-cw_out', 'apgd-adp-cw_v',
                                           'apgd-adp-cw_in-out', 'apgd-adp-cw_in-v', 'apgd-adp-cw_in-out-v',
                                           ]:
                    version = ood_attack_method
                    attacks_to_run = [ood_attack_method]
                    adversary = AutoAttack(model, norm='Linf', verbose=False, eps=args.attack_eps, version=version,
                                           attacks_to_run=attacks_to_run, num_in_classes=num_in_classes,
                                           num_out_classes=num_out_classes, num_v_classes=num_v_classes, data_type='out')
                    adversary.apgd.attack_other_in = attack_other_in
                    adversary.apgd.n_iter = aa_test_step
                    apgd_inputs = adversary.run_standard_evaluation(images, None, bs=curr_batch_size,
                                                               attack_all_emps=attack_all_emps,
                                                               best_loss=best_loss)
                    adv_inputs = pick_worst_images(model, num_in_classes, adv_inputs, apgd_inputs)
                elif ood_attack_method in ['apgd-adp-ce_in-targeted', 'apgd-adp-ce_in-out-targeted',
                                           'apgd-adp-ce_in-v-targeted', 'apgd-adp-ce_in-out-v-targeted',
                                           'apgd-adp-cw-targeted', 'apgd-adp-cw_in-targeted', 'apgd-adp-cw_in-out-targeted',
                                           'apgd-adp-cw_in-v-targeted', 'apgd-adp-cw_in-out-v-targeted']:
                    version = ood_attack_method
                    attacks_to_run = [version]
                    target_classes = model(images)[:, :num_in_classes].sort(dim=1)[1]
                    best_adv_x = images.clone()
                    best_out_ssp = torch.zeros((images.size(0),)).float().to(images.device) + 1.
                    for i in range(1, min(targets, target_classes.size(1) + 1)):
                        target_class = target_classes[:, -i]
                        adversary = AutoAttack(model, norm='Linf', verbose=False, eps=args.attack_eps, version=version,
                                               attacks_to_run=attacks_to_run, num_in_classes=num_in_classes,
                                               num_out_classes=num_out_classes, num_v_classes=num_v_classes,
                                               data_type='out')
                        adversary.apgd.attack_other_in = attack_other_in
                        adversary.apgd.n_iter = taa_test_step
                        adversary.apgd.n_restarts = 1
                        adv_x = adversary.run_standard_evaluation(images, None, bs=curr_batch_size,
                                                                  attack_all_emps=attack_all_emps,
                                                                  best_loss=best_loss, target=target_class)
                        with torch.no_grad():
                            prob = F.softmax(model(adv_x), dim=1)
                            out_ssp = prob[:, num_in_classes:].sum(dim=1)
                        best_ind = out_ssp < best_out_ssp
                        best_adv_x[best_ind] = adv_x[best_ind]
                        best_out_ssp[best_ind] = out_ssp[best_ind]
                    tapgd_inputs = best_adv_x
                    adv_inputs = pick_worst_images(model, num_in_classes, adv_inputs, tapgd_inputs)
                elif ood_attack_method == 'corrupt':
                    raise NotImplementedError
                elif ood_attack_method == 'adv_corrupt':
                    raise NotImplementedError
                elif ood_attack_method == 'clean':
                    adv_inputs = images
                else:
                    raise ValueError('un-supported ood_attack_method: {}'.format(ood_attack_method))

            with torch.no_grad():
                model.eval()
                logits = model(adv_inputs)
                probs = F.softmax(logits, dim=1)

            whole_scores, whole_preds = torch.max(probs, dim=1)
            miscls_in_indices = (whole_preds < num_in_classes)
            miscls_in_classes += miscls_in_indices.sum().item()

            if num_out_classes > 0:
                max_in_scores, _ = torch.max(probs[:, :num_in_classes], dim=1)
                sum_in_scores = torch.sum(probs[:, :num_in_classes], dim=1)

                max_out_scores, _ = torch.max(probs[:, num_in_classes:num_in_classes + num_out_classes], dim=1)
                sum_out_scores = torch.sum(probs[:, num_in_classes:num_in_classes + num_out_classes], dim=1)
            else:
                raise ValueError('num_out_classes should be > 0 !')

            # v_classes = probs.size(1) - num_in_classes - num_out_classes
            if num_v_classes > 0:
                miscls_v_indices = torch.logical_and((whole_preds >= num_in_classes + num_out_classes),
                                                     (whole_preds < num_in_classes + num_out_classes + num_v_classes))
                miscls_v_classes += miscls_v_indices.sum().item()
                max_v_scores, _ = torch.max(probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes], dim=1)
                sum_v_scores = torch.sum(probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes], dim=1)
            else:
                max_v_scores = torch.zeros((probs.size(0), )).to(device)
                sum_v_scores = torch.zeros((probs.size(0), )).to(device)

            for i in range(0, len(probs)):
                f_misc_scores.write(
                    "{},{},{},{},{},{}\n".format(max_in_scores[i].cpu().numpy(), sum_in_scores[i].cpu().numpy(),
                                                 max_out_scores[i].cpu().numpy(), sum_out_scores[i].cpu().numpy(),
                                                 max_v_scores[i].cpu().numpy(), sum_v_scores[i].cpu().numpy()))

            count += curr_batch_size
            cur_in_msps.append(max_in_scores)
            cur_out_msps.append(probs[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[0])
            cur_out_ssps.append(probs[:, num_in_classes:num_in_classes + num_out_classes].sum(dim=1))
            if num_v_classes > 0:
                cur_v_msps.append(probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes].max(dim=1)[0])
                cur_v_ssps.append(probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes].sum(dim=1))
            else:
                cur_v_msps.append(torch.zeros((probs.size(0), ), device=probs.device))
                cur_v_ssps.append(torch.zeros((probs.size(0),), device=probs.device))
            cur_data_logits.append(logits)
            # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time() - st))
            # print('batch:', j, 'msp:', max_in_scores.mean().item())
            # exit()

        cur_data_logits = torch.cat(cur_data_logits)
        np.save(saved_logits_file, cur_data_logits.cpu().numpy())
        f_misc_scores.close()
        saved_logits_files.append(saved_logits_file)
        all_in_msps[out_dataset] = torch.cat(cur_in_msps)
        all_out_msps[out_dataset] = torch.cat(cur_out_msps)
        all_out_ssps[out_dataset] = torch.cat(cur_out_ssps)
        all_v_msps[out_dataset] = torch.cat(cur_v_msps)
        all_v_ssps[out_dataset] = torch.cat(cur_v_ssps)
        all_inputs_num[out_dataset] = count
        all_miscls_in_classes[out_dataset] = miscls_in_classes
        all_miscls_v_classes[out_dataset] = miscls_v_classes

    return saved_logits_files, all_inputs_num, all_miscls_in_classes, all_miscls_v_classes, all_in_msps, all_out_msps, all_out_ssps, all_v_msps, all_v_ssps


def eval_on_signle_ood_dataset(id_scores, out_datasets, indiv_ood_score_dict, ts=[95], scoring_func='in_msp',
                               storage_device=torch.device('cpu')):
    indiv_auc = {}
    sum_auc = 0

    indiv_fprN = {}
    sum_of_fprN = {}
    indiv_tprN = {}
    sum_of_tprN = {}

    indiv_mean_score = {}
    sum_of_mean_score = 0

    mixing_ood_scores = []

    for out_dataset in out_datasets:
        cur_ood_socres = indiv_ood_score_dict[out_dataset]
        cur_ood_auroc = nn_util.auroc(id_scores, cur_ood_socres, scoring_func=scoring_func)
        indiv_auc[out_dataset] = cur_ood_auroc
        sum_auc += cur_ood_auroc

        for t in ts:
            fprN, _ = nn_util.fpr_at_tprN(id_scores, cur_ood_socres, TPR=t, scoring_func=scoring_func)
            if t not in indiv_fprN:
                indiv_fprN[t] = {}
                sum_of_fprN[t] = 0
            indiv_fprN[t][out_dataset] = fprN
            sum_of_fprN[t] += fprN

            tprN, _ = nn_util.tpr_at_tnrN(id_scores, cur_ood_socres, TNR=t, scoring_func=scoring_func)
            if t not in indiv_tprN:
                indiv_tprN[t] = {}
                sum_of_tprN[t] = 0
            indiv_tprN[t][out_dataset] = tprN
            sum_of_tprN[t] += tprN

        cur_mean_socre = cur_ood_socres.mean().item()
        indiv_mean_score[out_dataset] = cur_mean_socre
        sum_of_mean_score += cur_mean_socre
        mixing_ood_scores.append(cur_ood_socres)

    avg_auc = sum_auc / len(out_datasets)
    avg_ood_score = sum_of_mean_score / len(out_datasets)

    mixing_ood_scores = torch.cat(mixing_ood_scores)
    mixing_score = mixing_ood_scores.mean().item()
    mixing_auc = nn_util.auroc(id_scores, mixing_ood_scores, scoring_func=scoring_func)

    avg_fprN = {}
    mixing_fprN = {}
    for t in ts:
        avg_fprN[t] = sum_of_fprN[t] / len(out_datasets)
        mixing_fprN[t], _ = nn_util.fpr_at_tprN(id_scores, mixing_ood_scores, TPR=t, scoring_func=scoring_func)

    avg_tprN = {}
    mixing_tprN = {}
    for t in ts:
        avg_tprN[t] = sum_of_tprN[t] / len(out_datasets)
        mixing_tprN[t], _ = nn_util.tpr_at_tnrN(id_scores, mixing_ood_scores, TNR=t, scoring_func=scoring_func)

    return (avg_auc, avg_fprN, avg_tprN, avg_ood_score), (indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score), (
        mixing_auc, mixing_fprN, mixing_tprN, mixing_score)


def eval_ood_detection(model, num_in_classes, id_score_dict, socre_save_dir, out_datasets, args,
                       ood_attack_method_arrs=[['clean'], ], ts=[95], attack_other_in=False, attack_all_emps=False,
                       best_loss=True):
    storage_device = args.storage_device
    worst_ood_score_dict = {}
    for ood_attack_methods in ood_attack_method_arrs:
        st=time.time()
        saved_logits_files, inputs_num, miscls_in_classes, miscls_v_classes, in_msp_dict, out_msp_dict, out_ssp_dict, _, v_ssp_dict \
            = get_ood_scores(model, num_in_classes, socre_save_dir, out_datasets, args,
                             ood_attack_methods=ood_attack_methods, ood_batch_size=args.ood_batch_size,
                             attack_other_in=attack_other_in, attack_all_emps=attack_all_emps, best_loss=best_loss,
                             pgd_test_step=args.pgd_test_step, aa_test_step=args.aa_test_step,
                             taa_test_step=args.taa_test_step, targets=args.targets)
        v_and_out_minus_in_dict = {}
        v_and_out_dict = {}
        out_minus_in_dict = {}
        for temp_key, _ in out_ssp_dict.items():
            v_and_out_minus_in_dict[temp_key] = v_ssp_dict[temp_key] + out_ssp_dict[temp_key] - in_msp_dict[temp_key]
            v_and_out_dict[temp_key] = v_ssp_dict[temp_key] + out_ssp_dict[temp_key]
            out_minus_in_dict[temp_key] = out_ssp_dict[temp_key] - in_msp_dict[temp_key]

        for sc_key, ood_values in {'in_msp': [in_msp_dict, 'in_msp'],
                                   'v-out_ssp_minus_in_msp': [v_and_out_minus_in_dict, 'r_ssp'],
                                   'v-out_ssp': [v_and_out_dict, 'r_ssp'],
                                   'out_ssp_minus_in_msp': [out_minus_in_dict, 'r_ssp'],
                                   'out_msp': [out_msp_dict, 'r_ssp'],
                                   'out_ssp': [out_ssp_dict, 'r_ssp']}.items():
            if sc_key not in id_score_dict:
                continue
            con_f = id_score_dict[sc_key]
            print('=====> using {} as scoring function =====>'.format(sc_key))
            conf_t = ood_values[0]
            scoring_func = ood_values[1]
            (avg_auc, avg_fprN, avg_tprN, avg_ood_score), (indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score), (
                mixing_auc, mixing_fprN, mixing_tprN, mixing_score) \
                = eval_on_signle_ood_dataset(con_f, out_datasets, conf_t, ts=ts, scoring_func=scoring_func,
                                             storage_device=storage_device)
            total_inputs_num = 0
            sum_miscls_v_classes = 0
            sum_miscls_in_classes = 0
            for out_dataset in out_datasets:
                if out_dataset not in worst_ood_score_dict:
                    worst_ood_score_dict[out_dataset] = conf_t[out_dataset]
                else:
                    indcs = worst_ood_score_dict[out_dataset] < conf_t[out_dataset]
                    worst_ood_score_dict[out_dataset][indcs] = conf_t[out_dataset][indcs]

                total_inputs_num += inputs_num[out_dataset]
                sum_miscls_v_classes += miscls_v_classes[out_dataset]
                sum_miscls_in_classes += miscls_in_classes[out_dataset]
            print('----------------------------------------------------------------')
            print('With "{} (attacked)" OOD inputs, total miscls in-classes:{}(/{}), total miscls v-classes:{}(/{})'
                  .format(ood_attack_methods, sum_miscls_in_classes, inputs_num, sum_miscls_v_classes, inputs_num, ))
            print('Individual Performance: avg_auc:{}, avg_fprN:{}, avg_tprN:{}, avg_ood_score:{}'
                  .format(indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score))
            print('Avg Performance: indiv_auc:{}, indiv_fprN:{}, indiv_tprN:{}, indiv_mean_score:{}'
                  .format(avg_auc, avg_fprN, avg_tprN, avg_ood_score))
            print('Performance on Mixing: mixing_auc:{}, mixing_fprN:{}, mixing_tprN:{}, mixing_score:{}'.format(
                mixing_auc, mixing_fprN, mixing_tprN, mixing_score))
            print('eval time: {}s'.format(time.time() - st))
        print()


def get_all_data(test_loader, max_num=float('inf')):
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
        if len(y_test) >= max_num:
            return x_test, y_test
    return x_test, y_test


def filter_worst_cases(model, old_x, old_stat_indcs, new_x, new_stat_indcs, num_in_classes, batch_size=128):
    model.eval()

    new_adv_indcs = torch.logical_and(~old_stat_indcs, new_stat_indcs)
    both_adv_indcs = torch.logical_and(old_stat_indcs, new_stat_indcs)
    old_x[new_adv_indcs] = new_x[new_adv_indcs]

    temp_old_x = old_x[both_adv_indcs]
    temp_new_x = new_x[both_adv_indcs]

    num_examples = len(temp_old_x)
    # print('len of both_adv:', len(temp_old_x))
    for idx in range(0, num_examples, batch_size):
        st_idx = idx
        end_idx = min(idx + batch_size, num_examples)
        batch_old_x = temp_old_x[st_idx:end_idx]
        batch_new_x = temp_new_x[st_idx:end_idx]
        with torch.no_grad():
            old_outputs = F.softmax(model(batch_old_x), dim=1)
            old_scores, _ = torch.max(old_outputs[:, :num_in_classes], dim=1)
            new_outputs = F.softmax(model(batch_new_x), dim=1)
            new_scores, _ = torch.max(new_outputs[:, :num_in_classes], dim=1)
            indcs = new_scores > old_scores
            batch_old_x[indcs] = batch_new_x[indcs]
            # print('len of batch_old_x:', len(batch_old_x))
    old_stat_indcs = torch.logical_or(new_adv_indcs, both_adv_indcs)
    return old_x, old_stat_indcs


def check_adv_status(model, x, adv_x, y, num_in_classes, batch_size):
    model.eval()
    num_examples = len(x)
    org_incorr_indcs = []
    succ_pertub_indcs = []
    incorr_indcs = []
    corr = 0
    for idx in range(0, len(x), batch_size):
        st_idx = idx
        end_idx = min(idx + batch_size, num_examples)

        batch_x = x[st_idx:end_idx]
        batch_adv_x = adv_x[st_idx:end_idx]
        batch_y = y[st_idx:end_idx]
        with torch.no_grad():
            nat_output = model(batch_x)
            adv_output = model(batch_adv_x)
        nat_pred = torch.max(nat_output[:, :num_in_classes], dim=1)[1]
        adv_pred = torch.max(adv_output[:, :num_in_classes], dim=1)[1]
        succ_indcs = torch.logical_and(nat_pred == batch_y, adv_pred != batch_y)
        succ_pertub_indcs.append(succ_indcs)
        org_incorr_indcs.append(nat_pred != batch_y)
        incorr_indcs.append(adv_pred != batch_y)
        corr += (nat_pred == batch_y).sum().item()
    # print('corr acc:', corr / len(x))
    org_incorr_indcs = torch.cat(org_incorr_indcs)
    succ_pertub_indcs = torch.cat(succ_pertub_indcs)
    incorr_indcs = torch.cat(incorr_indcs)

    return org_incorr_indcs, succ_pertub_indcs, incorr_indcs


def get_target_classes(model, test_loader, num_in_classes, device=torch.device('cuda')):
    target_classes = []
    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        u = torch.arange(batch_x.shape[0])
        with torch.no_grad():
            logits = model(batch_x)
            temp_logits = logits.clone()
            temp_logits[u, batch_y] = -float('inf')
            logits_sorted, ind_sorted = temp_logits[:, :num_in_classes].sort(dim=1)  # small->larger
            target_classes.append(ind_sorted)
    target_classes = torch.cat(target_classes)
    return target_classes


def attack_id(model, test_loader, num_in_classes, args, version='standard-aa', attack_all_emps=False,
              best_loss=False, norm='Linf', num_ids=float('inf'), device=torch.device('cuda')):
    from autoattack import AutoAttack
    attack_eps = args.attack_eps
    num_out_classes = 0
    if 'num_out_classes' in args:
        num_out_classes = args.num_out_classes
    num_v_classes = 0
    if 'num_v_classes' in args:
        num_v_classes = args.num_v_classes
    model.eval()

    x_test, y_test = get_all_data(test_loader, max_num=num_ids)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    if version in ['ce', 'adp-ce_in', 'adp-ce_out', 'adp-ce_v', 'adp-ce_in-out', 'adp-ce_in-v', 'adp-ce_in-out-v',
                   'cw', 'adp-cw_in', 'adp-cw_out', 'adp-cw_v', 'adp-cw_in-out', 'adp-cw_in-v', 'adp-cw_in-out-v']:
        _, _, adv_x = pgd_attack_id(model, test_loader, num_in_classes, args, loss_str='ce',
                                    attack_all_emps=attack_all_emps, best_loss=best_loss, num_ids=num_ids)
        return x_test, y_test, adv_x
    elif version in ['apgd-ce', 'apgd-adp-ce_in', 'apgd-adp-ce_out', 'apgd-adp-ce_v', 'apgd-adp-ce_in-out', 'apgd-adp-ce_in-v', 'apgd-adp-ce_in-out-v',
                     'apgd-cw', 'apgd-adp-cw_in', 'apgd-adp-cw_out', 'apgd-adp-cw_v', 'apgd-adp-cw_in-out', 'apgd-adp-cw_in-v', 'apgd-adp-cw_in-out-v']:
        attacks_to_run = [version]
        adversary = AutoAttack(model, norm=norm, verbose=False, eps=attack_eps, version=version,
                               attacks_to_run=attacks_to_run, num_in_classes=num_in_classes,
                               num_out_classes=num_out_classes, num_v_classes=num_v_classes, data_type='in')
        if args.aa_test_step > adversary.apgd.n_iter:
            adversary.apgd.n_iter = args.aa_test_step
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size,
                                                         attack_all_emps=attack_all_emps,
                                                         best_loss=best_loss)
        return x_test, y_test, adv_complete
    elif version in ['apgd-adp-ce_in-targeted', 'apgd-adp-ce_in-out-targeted', 'apgd-adp-ce_in-out-v-targeted',
                     'apgd-adp-cw-targeted', 'apgd-adp-cw_in-targeted', 'apgd-adp-cw_in-out-targeted',
                     'apgd-adp-cw_in-out-v-targeted']:
        attacks_to_run = [version]
        target_classes = get_target_classes(model, test_loader, num_in_classes, device)
        best_adv_x = x_test.clone()
        worst_msp = torch.zeros((len(x_test), )).float().to(device)
        if 'targets' not in args:
            targets = 9
        else:
            targets = args.targets
        for i in range(1, min(targets, target_classes.size(1))):
            target_class = target_classes[:, -i]
            adversary = AutoAttack(model, norm=norm, verbose=False, eps=attack_eps, version=version,
                                   attacks_to_run=attacks_to_run, num_in_classes=num_in_classes,
                                   num_out_classes=num_out_classes, num_v_classes=num_v_classes, data_type='in')
            adversary.apgd.n_restarts = 1
            # if args.aa_test_step > adversary.apgd.n_iter:
            #     adversary.apgd.n_iter = args.aa_test_step
            adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size,
                                                             attack_all_emps=attack_all_emps,
                                                             best_loss=best_loss, target=target_class)
            org_incorr_indcs, succ_pertub_indcs, incorr_indcs \
                = check_adv_status(model, x_test, adv_complete, y_test, num_in_classes, args.batch_size)
            acc, _, _, msp, _, _, _, _, _ = nn_util.eval_from_data_with_out_classes(model, adv_complete, y_test,
                                                                                    args.batch_size, num_in_classes,
                                                                                    num_out_classes,
                                                                                    num_v_classes=num_v_classes)
            max_ind = msp[succ_pertub_indcs] > worst_msp[succ_pertub_indcs]
            temp_msp = worst_msp[succ_pertub_indcs] + 0.
            temp_msp[max_ind] = msp[succ_pertub_indcs][max_ind] + 0.
            worst_msp[succ_pertub_indcs] = temp_msp + 0.

            temp_adv_x = best_adv_x[succ_pertub_indcs] + 0.
            temp_adv_x[max_ind] = adv_complete[succ_pertub_indcs][max_ind] + 0.
            best_adv_x[succ_pertub_indcs] = temp_adv_x + 0.

        return x_test, y_test, best_adv_x
    elif version in ['standard-aa']:
        version = 'standard'
        adversary = AutoAttack(model, norm=norm, verbose=True, eps=attack_eps, version=version,
                               num_in_classes=num_in_classes, num_out_classes=num_out_classes,
                               num_v_classes=num_v_classes, data_type='in')
        adversary.apgd_targeted.n_target_classes = num_in_classes + num_out_classes + num_v_classes - 1
        adversary.fab.n_target_classes = num_in_classes + num_out_classes + num_v_classes - 1
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size, attack_all_emps=False,
                                                         best_loss=True)
        return x_test, y_test, adv_complete
    else:
        raise ValueError('un-supported attatck version: {}'.format(version))


def pgd_attack_id(model, test_loader, num_in_classes, args, loss_str='ce', attack_all_emps=False, best_loss=True,
                  num_ids=float('inf'), device=torch.device('cuda')):
    num_out_classes = 0
    if 'num_out_classes' in args:
        num_out_classes = args.num_out_classes
    num_v_classes = 0
    if 'num_v_classes' in args:
        num_v_classes = args.num_v_classes
    model.eval()
    nat_x = []
    adv_x = []
    y = []
    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if (i + 1) * len(batch_x) > num_ids:
            break
        if not attack_all_emps:
            ind = model(batch_x)[:, :num_in_classes].max(dim=1)[1] == batch_y
        else:
            ind = batch_y == batch_y
        x_to_fool = batch_x[ind]
        y_to_fool = batch_y[ind]
        if loss_str == 'fgsm':
            attacked_x = pgd.fgsm_attack(model, x_to_fool, y_to_fool, attack_eps=args.attack_eps, best_loss=best_loss)
        else:
            attacked_x = pgd.pgd_attack(model, x_to_fool, y_to_fool, args.pgd_test_step, args.attack_lr,
                                        args.attack_eps, loss_str=loss_str, num_in_classes=num_in_classes,
                                        num_out_classes=num_out_classes, num_v_classes=num_v_classes, data_type='in',
                                        best_loss=best_loss)
        batch_adv_x = batch_x.clone()
        batch_adv_x[ind] = attacked_x

        adv_x.append(batch_adv_x)
        nat_x.append(batch_x)
        y.append(batch_y)
    nat_x = torch.cat(nat_x)
    adv_x = torch.cat(adv_x)
    y = torch.cat(y)

    return nat_x, y, adv_x


def eval_in_and_out(model, test_loader, num_in_classes, args, device=torch.device('cuda'),
                    storage_device=torch.device('cpu')):
    socre_save_dir = args.save_socre_dir
    if not os.path.exists(socre_save_dir):
        os.makedirs(socre_save_dir)
    num_out_classes = 0
    if 'num_out_classes' in args:
        num_out_classes = args.num_out_classes
    num_v_classes = 0
    if 'num_v_classes' in args:
        num_v_classes = args.num_v_classes

    # clean IDs
    print('----------------------------------------------------------------')
    clean_id_misc_scores_file = os.path.join(socre_save_dir, 'clean_id_misc_scores.txt')
    clean_id_logits_file = os.path.join(socre_save_dir, 'clean_id_logits.npy')
    nat_id_acc, nat_miscls_out_cls, nat_miscls_v_cls, nat_id_msp, nat_id_corr_prob, nat_id_out_msp, nat_id_out_ssp, _, nat_id_v_ssp = \
        nn_util.eval_with_out_classes(model, test_loader, num_in_classes, num_out_classes, num_v_classes=num_v_classes,
                                      misc_score_file=clean_id_misc_scores_file, lotis_file = clean_id_logits_file)
    nat_id_mmsp = nat_id_msp.mean().item()
    nat_id_corr_mprob = nat_id_corr_prob.mean().item()
    nat_id_out_mmsp = nat_id_out_msp.mean().item()
    nat_id_out_mssp = nat_id_out_ssp.mean().item()
    nat_id_vout = nat_id_v_ssp + nat_id_out_ssp
    nat_id_m_vout = nat_id_vout.mean().item()
    nat_id_vout_in = nat_id_v_ssp + nat_id_out_ssp - nat_id_msp
    nat_id_m_vout_in = nat_id_vout_in.mean().item()
    print('Tetsing nat ID acc: {}, miscls into out classes: {}, miscls into v classes: {}, mean of in-msp: {}, '
          'mean of nat corr-prob: {}, mean of out-msp: {}, mean of out-ssp: {}, mean of (v_ssp + out-ssp): {}, '
          'mean of (v_ssp + out-ssp - in-msp): {}'.format(nat_id_acc, nat_miscls_out_cls, nat_miscls_v_cls, nat_id_mmsp,
                                                          nat_id_corr_mprob, nat_id_out_mmsp, nat_id_out_mssp,
                                                          nat_id_m_vout, nat_id_m_vout_in))
    print()

    # detecting OODs
    ts = [80, 85, 90, 95]
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        out_datasets = ['places365', 'svhn', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
    elif args.dataset == 'svhn':
        out_datasets = ['cifar10', 'cifar100', 'places365', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']

    print('=============================== OOD Detection performance =====================================')
    st = time.time()
    id_score_dict = {'in_msp': nat_id_msp, 'out_ssp_minus_in_msp': nat_id_out_ssp - nat_id_msp, 'out_ssp': nat_id_out_ssp}
    ood_attack_method_arrs = [['clean'],
                          ['adp-ce_in'], ['adp-ce_out'], ['adp-ce_in-out'],
                          ['adp-cw_in'], ['adp-cw_out'], ['adp-cw_in-out'],
                          ['apgd-adp-ce_in'], ['apgd-adp-ce_out'], ['apgd-adp-ce_in-out'],
                          ['apgd-adp-cw_in'], ['apgd-adp-cw_out'], ['apgd-adp-cw_in-out'],
                          ['apgd-adp-ce_in-targeted'], ['apgd-adp-ce_in-out-targeted'],
                          ['apgd-adp-cw_in-targeted'], ['apgd-adp-cw_in-out-targeted'],
                          ['clean', 'adp-ce_in-out', 'apgd-adp-cw_in-out-targeted'],
                          ]
    eval_ood_detection(model, num_in_classes, id_score_dict, socre_save_dir=socre_save_dir,
                       out_datasets=out_datasets, args=args, ood_attack_method_arrs=ood_attack_method_arrs, ts=ts,
                       attack_other_in=False, attack_all_emps=True, best_loss=True)
    print('evaluation time: {}s'.format(time.time() - st))
    print()
