import torch
import torch.nn.functional as F
from utils import nn_util
import numpy as np

def fgsm_attack(model, x, y, attack_eps=0.3, random_init=True, random_type='uniform', bn_type='eval', clamp=(0, 1),
                best_loss=False):
    attack_lr = attack_eps
    return pgd_attack(model, x, y, attack_step=1, attack_lr=attack_lr, attack_eps=attack_eps, random_init=random_init,
                      random_type=random_type, bn_type=bn_type, clamp=clamp, loss_str='ce', best_loss=best_loss)


def apgd_attack(model, x, y, attack_step=20, attack_eps=0.3, loss_str='ce', num_in_classes=10, num_out_classes=0,
                num_v_classes=0, best_loss=False):
    from autoattack import AutoAttack
    if loss_str in ['apgd-ce', 'apgd-adp-ce_out', 'apgd-adp-ce_in-out']:
        version = loss_str
        attacks_to_run = [loss_str]
        adversary = AutoAttack(model, norm='Linf', verbose=False, eps=attack_eps, version=version,
                               attacks_to_run=attacks_to_run, num_in_classes=num_in_classes,
                               num_out_classes=num_out_classes, num_v_classes=num_v_classes, data_type='in')
        adversary.apgd.n_iter = attack_step
        adversary.apgd.n_restarts = 1
        adv_ood = adversary.run_standard_evaluation(x, y, bs=len(x), attack_all_emps=False, best_loss=best_loss)
        return adv_ood
    else:
        raise ValueError('un-supported loss_str: {}'.format(loss_str))


def pgd_attack_batch(model, x, y, batch_size, attack_step, attack_lr=0.003, attack_eps=0.3, random_init=True,
                     random_type='uniform', bn_type='eval', clamp=(0, 1), loss_str='ce', num_in_classes=10,
                     attack_other_in=False, num_out_classes=0, num_v_classes=0, data_type='', y_o=None,
                     best_loss=False):
    num_examples = len(x)
    adv_x = torch.tensor([]).to(x.device)
    for idx in range(0, len(x), batch_size):
        st_idx = idx
        end_idx = min(idx + batch_size, num_examples)
        batch_x = x[st_idx:end_idx]
        batch_y = y[st_idx:end_idx]
        batch_adv_x = pgd_attack(model, batch_x, batch_y, attack_step, attack_lr=attack_lr, attack_eps=attack_eps,
                                 random_init=random_init, random_type=random_type,
                                 bn_type=bn_type, clamp=clamp, loss_str=loss_str, num_in_classes=num_in_classes,
                                 attack_other_in=attack_other_in,
                                 num_out_classes=num_out_classes, num_v_classes=num_v_classes, data_type=data_type,
                                 y_o=y_o, best_loss=best_loss)
        adv_x=torch.cat((adv_x, batch_adv_x), 0)
    return adv_x



def pgd_attack(model, x, y, attack_step, attack_lr=0.003, attack_eps=0.3, random_init=True, random_type='uniform',
               bn_type='eval', clamp=(0, 1), loss_str='ce', num_in_classes=10, attack_other_in=False,
               num_out_classes=0, num_v_classes=0, data_type='', y_o=None, best_loss=False):
    if bn_type == 'eval':
        model.eval()
    elif bn_type == 'train':
        model.train()
    else:
        raise ValueError('error bn_type: {0}'.format(bn_type))
    x_adv = x.clone().detach()
    if random_init:
        # Flag to use random initialization
        if random_type == 'gussian':
            x_adv = x_adv + 0.001 * torch.randn(x.shape, device=x.device)
        elif random_type == 'uniform':
            # x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * attack_eps
            random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-attack_eps, attack_eps).to(x_adv.device)
            x_adv = x_adv + random_noise
        else:
            raise ValueError('error random noise type: {0}'.format(random_type))

    if best_loss:
        adv_best = x.detach().clone()
        loss_best = torch.ones([x.shape[0]]).to(x.device) * (-float('inf'))
    for i in range(attack_step):
        x_adv.requires_grad = True

        model.zero_grad()
        adv_logits = model(x_adv)

        # Untargeted attacks - gradient ascent
        if loss_str == 'ce':
            # loss = F.cross_entropy(adv_logits, y, reduction='none')
            if len(y.size()) == 2:
                loss = nn_util.cross_entropy_soft_target(adv_logits, y, reduction='none')
            else:
                loss = F.cross_entropy(adv_logits, y, reduction='none')
        elif loss_str == 'ce_in-out':
            assert num_out_classes > 0 and num_v_classes == 0
            in_loss = F.cross_entropy(adv_logits, y, reduction='none')
            y_out = adv_logits[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[1] + num_in_classes
            out_loss = F.cross_entropy(adv_logits, y_out, reduction='none')
            loss = in_loss + out_loss
        elif loss_str == 'ce-y_in-y_o':
            assert y_o is not None and len(y_o.size())==1
            assert num_out_classes > 0 and num_v_classes == 0
            in_loss = F.cross_entropy(adv_logits, y, reduction='none')
            out_loss = F.cross_entropy(adv_logits, y_o, reduction='none')
            loss = in_loss + out_loss
        elif loss_str == 'adp-ce_in':
            loss = adaptive_ce_loss(adv_logits, y, num_in_classes=num_in_classes, attack_in=True,
                                    num_out_classes=num_out_classes, attack_out=False, num_v_classes=num_v_classes,
                                    attack_v=False, data_type=data_type, reduction='none')
        elif loss_str == 'adp-ce_v':
            loss = adaptive_ce_loss(adv_logits, y, num_in_classes=num_in_classes, attack_in=False,
                                    num_out_classes=num_out_classes, attack_out=False, num_v_classes=num_v_classes,
                                    attack_v=True, data_type=data_type, reduction='none')
        elif loss_str == 'adp-ce_in-v':
            loss = adaptive_ce_loss(adv_logits, y, num_in_classes=num_in_classes, attack_in=True,
                                    num_out_classes=num_out_classes, attack_out=False, num_v_classes=num_v_classes,
                                    attack_v=True, data_type=data_type, reduction='none')
        elif loss_str == 'adp-ce_out':
            loss = adaptive_ce_loss(adv_logits, y, num_in_classes=num_in_classes, attack_in=False,
                                    num_out_classes=num_out_classes, attack_out=True, num_v_classes=num_v_classes,
                                    attack_v=False, data_type=data_type, reduction='none')
        elif loss_str == 'adp-ce_in-out':
            loss = adaptive_ce_loss(adv_logits, y, num_in_classes=num_in_classes, attack_in=True,
                                    num_out_classes=num_out_classes, attack_out=True, num_v_classes=num_v_classes,
                                    attack_v=False, data_type=data_type, reduction='none')
        elif loss_str == 'adp-ce_in-out-v':
            loss = adaptive_ce_loss(adv_logits, y, num_in_classes=num_in_classes, attack_in=True,
                                    num_out_classes=num_out_classes, attack_out=True, num_v_classes=num_v_classes,
                                    attack_v=True, data_type=data_type, reduction='none')
        elif loss_str == 'cw':
            loss = cw_loss(adv_logits, y, num_in_classes=num_in_classes, data_type=data_type, reduction='none')
        elif loss_str == 'adp-cw_in':
            loss = adaptive_cw_loss(adv_logits, y, num_in_classes=num_in_classes, attack_other_in=attack_other_in,
                                    num_out_classes=num_out_classes, attack_out=False, num_v_classes=num_v_classes,
                                    attack_v=False, data_type=data_type, reduction='none')
        elif loss_str == 'adp-cw_v':
            loss = adaptive_cw_loss(adv_logits, y, num_in_classes=num_in_classes, attack_in=False,
                                    attack_other_in=attack_other_in, num_out_classes=num_out_classes, attack_out=False,
                                    num_v_classes=num_v_classes, attack_v=True, data_type=data_type, reduction='none')
        elif loss_str == 'adp-cw_in-out':
            loss = adaptive_cw_loss(adv_logits, y, num_in_classes=num_in_classes, attack_other_in=attack_other_in,
                                    num_out_classes=num_out_classes, attack_out=True, num_v_classes=num_v_classes,
                                    attack_v=False, data_type=data_type, reduction='none')
        elif loss_str == 'adp-cw_out':
            loss = adaptive_cw_loss(adv_logits, y, num_in_classes=num_in_classes, attack_in=False,
                                    attack_other_in=attack_other_in, num_out_classes=num_out_classes, attack_out=True,
                                    num_v_classes=num_v_classes, attack_v=False, data_type=data_type, reduction='none')
        elif loss_str == 'adp-cw_in-v':
            loss = adaptive_cw_loss(adv_logits, y, num_in_classes=num_in_classes, attack_other_in=attack_other_in,
                                    num_out_classes=num_out_classes, attack_out=False, num_v_classes=num_v_classes,
                                    attack_v=True, data_type=data_type, reduction='none')
        elif loss_str == 'adp-cw_in-out-v':
            loss = adaptive_cw_loss(adv_logits, y, num_in_classes=num_in_classes, attack_other_in=attack_other_in,
                                    num_out_classes=num_out_classes, attack_out=True, num_v_classes=num_v_classes,
                                    attack_v=True, data_type=data_type, reduction='none')
        elif loss_str == 'dlr':
            loss = dlr_loss(adv_logits, y, num_in_classes=num_in_classes, data_type=data_type, reduction='none')
        elif loss_str == 'adp-dlr_in':
            loss = adaptive_dlr_loss(adv_logits, y, num_in_classes=num_in_classes, attack_other_in=attack_other_in,
                                     num_out_classes=num_out_classes, attack_out=False, num_v_classes=num_v_classes,
                                     attack_v=False, data_type=data_type, reduction='none')
        elif loss_str == 'adp-dlr_in-out':
            loss = adaptive_dlr_loss(adv_logits, y, num_in_classes=num_in_classes, attack_other_in=attack_other_in,
                                     num_out_classes=num_out_classes, attack_out=True, num_v_classes=num_v_classes,
                                     attack_v=False, data_type=data_type, reduction='none')
        elif loss_str == 'adp-dlr_in-v':
            loss = adaptive_dlr_loss(adv_logits, y, num_in_classes=num_in_classes, attack_other_in=attack_other_in,
                                     num_out_classes=num_out_classes, attack_out=False, num_v_classes=num_v_classes,
                                     attack_v=True, data_type=data_type, reduction='none')
        elif loss_str == 'kl':
            criterion_kl = torch.nn.KLDivLoss(reduction=False)
            logits = model(x)
            loss = criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(logits, dim=1))
        else:
            raise ValueError('un-supported adv loss:'.format(loss_str))

        loss.mean().backward()
        loss = loss.detach()
        grad = x_adv.grad.detach()
        grad = grad.sign()
        x_adv = x_adv.detach()
        x_adv = x_adv + attack_lr * grad

        # Projection
        x_adv = x + torch.clamp(x_adv - x, min=-attack_eps, max=attack_eps)
        x_adv = torch.clamp(x_adv, *clamp)
        if best_loss:
            ind_curr = loss > loss_best
            adv_best[ind_curr] = x_adv[ind_curr] + 0.
            loss_best[ind_curr] = loss[ind_curr] + 0.
    if best_loss:
        return adv_best
    return x_adv


def pgd_attack_targeted(model, x, y, y_target, attack_step, attack_lr=0.003, attack_eps=0.3, random_init=True,
                        random_type='uniform', bn_type='eval', clamp=(0, 1), loss_str='ce-targeted',
                        num_in_classes=10, attack_other_in=False, num_out_classes=0, num_v_classes=0, data_type='',
                        best_loss=False):
    if bn_type == 'eval':
        model.eval()
    elif bn_type == 'train':
        model.train()
    else:
        raise ValueError('error bn_type: {0}'.format(bn_type))
    x_adv = x.clone().detach()
    if random_init:
        # Flag to use random initialization
        if random_type == 'gussian':
            x_adv = x_adv + 0.001 * torch.randn(x.shape, device=x.device)
        elif random_type == 'uniform':
            # x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * attack_eps
            random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-attack_eps, attack_eps).to(x_adv.device)
            x_adv = x_adv + random_noise
        else:
            raise ValueError('error random noise type: {0}'.format(random_type))
    if best_loss:
        adv_best = x.detach().clone()
        loss_best = torch.ones([x.shape[0]]).to(x.device) * (-float('inf'))
    for i in range(attack_step):
        x_adv.requires_grad = True

        model.zero_grad()
        adv_logits = model(x_adv)

        # Untargeted attacks - gradient ascent
        if loss_str == 'ce-targeted':
            loss = -F.cross_entropy(adv_logits, y_target, reduction='none')
        elif loss_str == 'cw-targeted':
            loss = cw_loss_targeted(adv_logits, y, y_target, data_type, reduction='none')
        elif loss_str == 'adp-cw_in-targeted':
            loss = adaptive_cw_loss_targeted(adv_logits, y, y_target, num_in_classes, attack_other_in, num_out_classes,
                                             False, num_v_classes, False, data_type, reduction='none')
        elif loss_str == 'adp-cw_in-out-targeted':
            loss = adaptive_cw_loss_targeted(adv_logits, y, y_target, num_in_classes, attack_other_in, num_out_classes,
                                             True, num_v_classes, False, data_type, reduction='none')
        elif loss_str == 'adp-cw_in-v-targeted':
            loss = adaptive_cw_loss_targeted(adv_logits, y, y_target, num_in_classes, attack_other_in, num_out_classes,
                                             False, num_v_classes, True, data_type, reduction='none')
        elif loss_str == 'dlr-targeted':
            loss = dlr_loss_targeted(adv_logits, y, y_target, data_type, reduction='none')
        elif loss_str == 'adp-dlr_in-targeted':
            loss = adaptive_dlr_loss_targeted(adv_logits, y, y_target, num_in_classes, attack_other_in, num_out_classes,
                                             False, num_v_classes, False, data_type, reduction='none')
        elif loss_str == 'adp-dlr_in-out-targeted':
            loss = adaptive_dlr_loss_targeted(adv_logits, y, y_target, num_in_classes, attack_other_in, num_out_classes,
                                             True, num_v_classes, False, data_type, reduction='none')
        elif loss_str == 'adp-dlr_in-v-targeted':
            loss = adaptive_dlr_loss_targeted(adv_logits, y, y_target, num_in_classes, attack_other_in, num_out_classes,
                                             False, num_v_classes, True, data_type, reduction='none')
        else:
            raise ValueError('un-supported adv loss:'.format(loss_str))

        loss.mean().backward()
        loss=loss.detach()
        grad = x_adv.grad.detach()
        grad = grad.sign()
        x_adv = x_adv.detach()
        x_adv = x_adv + attack_lr * grad

        # Projection
        x_adv = x + torch.clamp(x_adv - x, min=-attack_eps, max=attack_eps)
        x_adv = torch.clamp(x_adv, *clamp)

        if best_loss:
            ind_curr = loss > loss_best
            adv_best[ind_curr] = x_adv[ind_curr] + 0.
            loss_best[ind_curr] = loss[ind_curr] + 0.

    if best_loss:
        return adv_best
    return x_adv


def dlr_loss_targeted(logits, y, y_target, data_type, reduction='mean'):
    u = torch.arange(logits.shape[0])
    logits_sorted, _ = logits.sort(dim=1)
    if data_type == 'in':
        losses = -(logits[u, y] - logits[u, y_target]) / (
                logits_sorted[:, -1] - .5 * (logits_sorted[:, -3] + logits_sorted[:, -4]) + 1e-12)
    elif data_type == 'out':
        with torch.no_grad():
            temp_logits = logits.clone()
            temp_logits[u, y_target] = -float('inf')
            other_ind = temp_logits.max(dim=1)[1]
        other_logit = logits[u, other_ind]
        losses = (logits[u, y_target] - other_logit) / (
                logits_sorted[:, -1] - .5 * (logits_sorted[:, -3] + logits_sorted[:, -4]) + 1e-12)
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError('un-supported reduction: {}'.format(reduction))


def cw_loss_targeted(logits, y, y_target, data_type, reduction='mean'):
    u = torch.arange(logits.shape[0])
    if data_type == 'in':
        losses = logits[u, y_target] - logits[u, y]
    elif data_type == 'out':
        with torch.no_grad():
            temp_logits = logits.clone()
            temp_logits[u, y_target] = -float('inf')
            other_ind = temp_logits.max(dim=1)[1]
        other_logit = logits[u, other_ind]
        losses = logits[u, y_target] - other_logit
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError('un-supported reduction: {}'.format(reduction))


def adaptive_dlr_loss_targeted(logits, y, y_target, num_in_classes, attack_other_in, num_out_classes, attack_out,
                               num_v_classes, attack_v, data_type, reduction = 'mean'):
    assert logits.size(1) == num_in_classes + num_out_classes + num_v_classes
    assert y_target < num_in_classes
    u = torch.arange(logits.shape[0])
    if data_type=='in':
        logits_sorted, _ = logits.sort(dim=1)
        losses = -(logits[u, y] - logits[u, y_target]) / (
                    logits_sorted[:, -1] - .5 * (logits_sorted[:, -3] + logits_sorted[:, -4]) + 1e-12)
    elif data_type=='out':
        max_out_logit = 0
        if num_out_classes > 0 and attack_out:
            max_out_logit = logits[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[0]
        max_v_logit = 0
        if num_v_classes > 0 and attack_v:
            max_v_logit = logits[:, num_in_classes + num_out_classes:].max(dim=1)[0]

        other_in_logit = 0
        if attack_other_in == True:
            with torch.no_grad():
                temp_logits = logits.clone()
                temp_logits[u, y_target] = -float('inf')
                other_in_ind = temp_logits[:, :num_in_classes].max(dim=1)[1]
            other_in_logit = logits[u, other_in_ind]

        logits_sorted, _ = logits.sort(dim=1)
        losses = (logits[u, y_target] - other_in_logit - max_out_logit - max_v_logit) / (
                logits_sorted[:, -1] - .5 * (logits_sorted[:, -3] + logits_sorted[:, -4]) + 1e-12)
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError('un-supported reduction: {}'.format(reduction))


def adaptive_cw_loss_targeted(logits, y, y_target, num_in_classes, attack_other_in, num_out_classes, attack_out,
                              num_v_classes, attack_v, data_type, reduction='mean'):
    assert logits.size(1) == num_in_classes + num_out_classes + num_v_classes
    assert y_target < num_in_classes
    u = torch.arange(logits.shape[0])
    if data_type == 'in':
        logits_sorted, _ = logits.sort(dim=1)
        losses = logits[u, y_target] - logits[u, y]
    elif data_type == 'out':
        max_out_logit = 0
        if num_out_classes > 0 and attack_out:
            max_out_logit = logits[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[0]
        max_v_logit = 0
        if num_v_classes > 0 and attack_v:
            max_v_logit = logits[:, num_in_classes + num_out_classes:].max(dim=1)[0]
        other_in_logit = 0
        if attack_other_in == True:
            with torch.no_grad():
                temp_logits = logits.clone()
                temp_logits[u, y_target] = -float('inf')
                other_in_ind = temp_logits[:, :num_in_classes].max(dim=1)[1]
            other_in_logit = logits[u, other_in_ind]

        losses = logits[u, y_target] - other_in_logit - max_out_logit - max_v_logit
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))
    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError('un-supported reduction: {}'.format(reduction))


def eval_pgdadv_with_out_classes(model, test_loader, attack_step, attack_lr, attack_eps, num_in_classes,
                                 attack_other_in, num_out_classes, num_v_classes=0, data_type='in', loss_str='ce',
                                 best_loss=False, misc_score_file='', lotis_file=''):
    model.eval()
    correct = 0
    total = 0
    miscls_out_cls = 0
    miscls_v_cls = 0
    all_corr_prob = []
    all_in_msp = []
    all_out_msp = []
    all_out_ssp = []
    all_v_msp = []
    all_v_ssp = []
    all_logits = []

    if misc_score_file != '':
        f_misc_score = open(misc_score_file, 'w')

    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x = batch_x.cuda(non_blocking=True)
        batch_y = batch_y.cuda(non_blocking=True)

        adv_batch_x = pgd_attack(model, batch_x, batch_y, attack_step, attack_lr, attack_eps,
                                 num_in_classes=num_in_classes, attack_other_in=attack_other_in,
                                 num_out_classes=num_out_classes, num_v_classes=num_v_classes, data_type=data_type,
                                 loss_str=loss_str, best_loss=best_loss)
        # compute output
        with torch.no_grad():
            logits = model(adv_batch_x)
            probs = F.softmax(logits, dim=1)

        # num_v_classes = probs.size(1) - num_in_classes - num_out_classes
        in_msp, in_preds = torch.max(probs[:, :num_in_classes], dim=1)
        correct_indcs = in_preds == batch_y
        correct += correct_indcs.sum().item()
        all_corr_prob.append(in_msp[correct_indcs])
        all_in_msp.append(in_msp)

        out_msp = probs[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[0]
        out_ssp = probs[:, num_in_classes:num_in_classes + num_out_classes].sum(dim=1)
        if num_v_classes>0:
            v_msp = probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes].max(dim=1)[0]
            v_ssp = probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes].sum(dim=1)
        else:
            v_msp = torch.zeros((probs.size(0),), device=probs.device)
            v_ssp = torch.zeros((probs.size(0),), device=probs.device)
        all_out_msp.append(out_msp)
        all_out_ssp.append(out_ssp)
        all_v_msp.append(v_msp)
        all_v_ssp.append(v_ssp)

        whole_probs, whole_preds = torch.max(probs, dim=1)
        if num_out_classes > 0:
            miscls_out_indcs = torch.logical_and((whole_preds < num_in_classes + num_out_classes),
                                                 (whole_preds >= num_in_classes))
            miscls_out_cls += (miscls_out_indcs).sum().item()
        if num_v_classes > 0:
            miscls_v_indcs = torch.logical_and((whole_preds >= num_in_classes + num_out_classes),
                                               (whole_preds < probs.size(1)))
            miscls_v_cls += (miscls_v_indcs).sum().item()

        total += batch_x.size(0)
        if misc_score_file != '':
            in_ssp = torch.sum(probs[:, :num_in_classes], dim=1)
            for i in range(0, len(probs)):
                f_misc_score.write("{}, {}, {}, {}, {}, {}\n".format(in_msp[i].cpu().numpy(), in_ssp[i].cpu().numpy(),
                                                                     out_msp[i].cpu().numpy(), out_ssp[i].cpu().numpy(),
                                                                     v_msp[i].cpu().numpy(), v_ssp[i].cpu().numpy()))
                
        if lotis_file != '':
            all_logits.append(logits)
    all_in_msp = torch.cat(all_in_msp)
    all_corr_prob = torch.cat(all_corr_prob)
    all_out_msp = torch.cat(all_out_msp)
    all_out_ssp = torch.cat(all_out_ssp)
    all_v_msp = torch.cat(all_v_msp)
    all_v_ssp = torch.cat(all_v_ssp)
    acc = (float(correct) / total)
    if misc_score_file != '':
        f_misc_score.close()
    if lotis_file != '':
        all_logits = torch.cat(all_logits)
        np.save(lotis_file, all_logits.cpu().numpy())

    return acc, miscls_out_cls, miscls_v_cls, all_in_msp, all_corr_prob, all_out_msp, all_out_ssp, all_v_msp, all_v_ssp


def eval_pgdadv(model, test_loader, attack_step, attack_lr, attack_eps, num_in_classes, attack_other_in, num_v_classes,
                data_type='in', loss_str='ce', best_loss=False, misc_score_file='', lotis_file=''):
    model.eval()
    corr = 0
    total = 0
    located_in_vcls = 0
    located_in_vcls_corr = 0
    all_corr_prob = []
    all_in_msp = []
    all_v_ssp = []
    all_v_msp = []
    all_logits = []

    if misc_score_file != '':
        f_misc_score = open(misc_score_file, 'w')

    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x = batch_x.cuda(non_blocking=True)
        batch_y = batch_y.cuda(non_blocking=True)

        adv_batch_x = pgd_attack(model, batch_x, batch_y, attack_step, attack_lr, attack_eps, loss_str=loss_str,
                                 num_in_classes=num_in_classes, attack_other_in=attack_other_in, num_out_classes=0,
                                 num_v_classes=num_v_classes, data_type=data_type, best_loss=best_loss)

        # compute output
        with torch.no_grad():
            logits = model(adv_batch_x)
            probs = F.softmax(logits, dim=1)

        in_msp, in_preds = torch.max(probs[:, :num_in_classes], dim=1)
        corr_indices = in_preds == batch_y
        corr += corr_indices.sum().item()
        all_corr_prob.append(in_msp[corr_indices])
        all_in_msp.append(in_msp)
        if num_v_classes > 0:
            v_ssp = probs[:, num_in_classes:num_in_classes + num_v_classes].sum(dim=1)
            v_msp = probs[:, num_in_classes:num_in_classes + num_v_classes].max(dim=1)[0]
        else:
            v_msp = torch.zeros((probs.size(0),), device=probs.device)
            v_ssp = torch.zeros((probs.size(0),), device=probs.device)
        all_v_ssp.append(v_ssp)
        all_v_msp.append(v_msp)

        _, whole_preds = torch.max(probs, dim=1)
        located_in_vclass_indices = torch.logical_and((whole_preds >= num_in_classes),
                                                      (whole_preds < num_in_classes + num_v_classes))
        located_in_vcls += located_in_vclass_indices.sum().item()
        located_in_vcls_corr += (
            torch.logical_and(corr_indices, located_in_vclass_indices)).sum().item()
        total += batch_y.size(0)

        if misc_score_file != '':
            in_ssp = torch.sum(probs[:, :num_in_classes], dim=1)
            for i in range(0, len(probs)):
                f_misc_score.write("{},{},{},{}\n".format(in_msp[i].cpu().numpy(), in_ssp[i].cpu().numpy(),
                                                          v_msp[i].cpu().numpy(), v_ssp[i].cpu().numpy()))
        if lotis_file != '':
            all_logits.append(logits)

    all_corr_prob = torch.cat(all_corr_prob)
    all_in_msp = torch.cat(all_in_msp)
    all_v_ssp = torch.cat(all_v_ssp)
    all_v_msp = torch.cat(all_v_msp)
    acc = (float(corr) / total)
    if misc_score_file != '':
        f_misc_score.close()
    if lotis_file != '':
        all_logits = torch.cat(all_logits)
        np.save(lotis_file, all_logits.cpu().numpy())

    return acc, located_in_vcls, located_in_vcls_corr, all_in_msp, all_corr_prob, all_v_msp, all_v_ssp


def OE_loss(logits):
    logits_dim = logits.size()[1]
    if logits_dim == 0:
        return torch.tensor(0).to(logits.device)
    return -(logits_dim.mean(1) - torch.logsumexp(logits_dim, dim=1)).mean()


def adaptive_ce_loss(logits, y, num_in_classes=10, attack_in=True, num_out_classes=0, attack_out=True, num_v_classes=0,
                     attack_v=False, reduction='mean', data_type='in'):
    assert attack_in | attack_out | attack_v == True
    assert logits.size(1) == num_in_classes + num_out_classes + num_v_classes
    out_loss = 0
    if num_out_classes > 0 and attack_out:
        out_ind = logits[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[1] + num_in_classes
        out_loss = F.cross_entropy(logits, out_ind, reduction='none')
    v_loss = 0
    if num_v_classes > 0 and attack_v:
        v_ind = logits[:, num_in_classes + num_out_classes:].max(dim=1)[1] + num_in_classes + num_out_classes
        v_loss = F.cross_entropy(logits, v_ind, reduction='none')

    in_loss = 0
    if data_type == 'in':
        if attack_in:
            in_loss = F.cross_entropy(logits, y, reduction='none')
        losses = in_loss + out_loss + v_loss
    elif data_type == 'out':
        if attack_in:
            in_pred = logits[:, :num_in_classes].max(dim=1)[1]
            in_loss = -F.cross_entropy(logits, in_pred, reduction='none')
        losses = in_loss + out_loss + v_loss
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError('un-supported reduction: {}'.format(reduction))


def adaptive_cw_loss(logits, y, num_in_classes=10, attack_in=True, attack_other_in=False, num_out_classes=0,
                     attack_out=True, num_v_classes=0, attack_v=False, reduction='mean', data_type='in'):
    assert logits.size(1) == num_in_classes + num_out_classes + num_v_classes
    # num_v_classes = logits.size(1) - num_in_classes - num_out_classes
    max_out_logit = 0
    if num_out_classes > 0 and attack_out:
        max_out_logit = logits[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[0]
    max_v_logit = 0
    if num_v_classes > 0 and attack_v:
        max_v_logit = logits[:, num_in_classes + num_out_classes:].max(dim=1)[0]

    if data_type == 'in':
        indcs = torch.arange(logits.size(0))
        with torch.no_grad():
            temp_logits = logits.clone()
            temp_logits[indcs, y] = -float('inf')
            other_in_ind = temp_logits[:, :num_in_classes].max(dim=1)[1]
        other_in_logit = logits[indcs, other_in_ind]
        corr_logit = logits[indcs, y]
        losses = other_in_logit - corr_logit - max_out_logit - max_v_logit
    elif data_type == 'out':
        assert (attack_in | attack_out | attack_v) == True
        # max_in_logits = logits[:, :num_in_classes].max(dim=1)[0]
        top_logits, top_ind = logits[:, :num_in_classes].topk(k=2, dim=1)
        if attack_in:
            max_in_logit = top_logits[:, 0]
        else:
            max_in_logit = 0
        second_in_logit = 0
        if attack_other_in == True:
            second_in_logit = top_logits[:, 1]
        losses = max_in_logit - second_in_logit - max_out_logit - max_v_logit
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError('un-supported reduction: {}'.format(reduction))


def cw_loss(logits, y, num_in_classes=10, data_type='in', reduction='mean'):
    u = torch.arange(logits.size(0))
    if data_type == 'in':
        with torch.no_grad():
            temp_logits = logits.clone()
            temp_logits[u, y] = -float('inf')
            other_ind = temp_logits.max(dim=1)[1]
        losses = logits[u, other_ind] - logits[u, y]
    elif data_type == 'out':
        max_in_logits, max_in_ind = logits[:, :num_in_classes].max(dim=1)
        with torch.no_grad():
            temp_logits = logits.clone()
            temp_logits[u, max_in_ind] = -float('inf')
            other_ind = temp_logits.max(dim=1)[1]
        losses = max_in_logits - logits[u, other_ind]
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError('un-supported reduction: {}'.format(reduction))


def adaptive_dlr_loss(logits, y, num_in_classes=10, attack_other_in=False, num_out_classes=0, attack_out=True,
                      num_v_classes=0, attack_v=False, reduction='mean', data_type='in'):
    assert logits.size(1) == num_in_classes + num_out_classes + num_v_classes
    max_out_logit = 0
    if num_out_classes > 0 and attack_out:
        max_out_logit = logits[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[0]
    max_v_logit = 0
    if num_v_classes > 0 and attack_v:
        max_v_logit = logits[:, num_in_classes + num_out_classes:].max(dim=1)[0]
    whole_confs, _ = logits.topk(k=3, dim=1)
    whole_in_max = whole_confs[:, 0]
    whole_in_3th = whole_confs[:, 2]
    if data_type == 'in':
        indcs = torch.arange(logits.size(0))
        with torch.no_grad():
            temp_logits = logits.clone()
            temp_logits[indcs, y] = -float('inf')
            other_in_ind = temp_logits[:, :num_in_classes].max(dim=1)[1]
        other_in_logit = logits[indcs, other_in_ind]
        corr_logit = logits[indcs, y]
        losses = (other_in_logit - corr_logit - max_out_logit - max_v_logit) / (whole_in_max - whole_in_3th + 1e-12)
    elif data_type == 'out':
        # max_in_logit = logits[:, :num_in_classes].max(dim=1)[0]
        top_logits, top_ind = logits[:, :num_in_classes].topk(k=2, dim=1)
        max_in_logit = top_logits[:, 0]
        second_in_logit = 0
        if attack_other_in == True:
            second_in_logit = top_logits[:, 1]
        losses = (max_in_logit - second_in_logit - max_out_logit - max_v_logit) / (whole_in_max - whole_in_3th + 1e-12)
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError('un-supported reduction: {}'.format(reduction))


def dlr_loss(logits, y, num_in_classes=10, data_type='in', reduction='mean'):
    if data_type == 'in':
        logits_sorted, ind_sorted = logits[:, :num_in_classes].sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(logits.shape[0])
        losses = -(logits[u, y] - logits_sorted[:, -2] * ind - logits_sorted[:, -1] * (1. - ind)) / \
               (logits_sorted[:, -1] - logits_sorted[:, -3] + 1e-12)
    elif data_type == 'out':
        u = torch.arange(logits.size(0))
        logits_sorted, _ = logits[:, :num_in_classes].sort(dim=1)
        max_in_logit, max_in_ind = logits[:, :num_in_classes].max(dim=1)
        with torch.no_grad():
            temp_logits = logits.clone()
            temp_logits[u, max_in_ind] = -float('inf')
            other_ind = temp_logits.max(dim=1)[1]
        losses = (max_in_logit -logits[u, other_ind]) / (logits_sorted[:, -1] - logits_sorted[:, -3] + 1e-12)
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError('un-supported reduction: {}'.format(reduction))


def apgd_attack_ood_misc(model, x, y, num_in_classes, num_out_classes=0, num_v_classes=0, attack_step=20,
                         attack_eps=0.3, loss_str='', best_loss=False):
    from autoattack import AutoAttack
    if loss_str in ['apgd-oe', 'apgd-ce', 'apgd-adp-ce_out', 'apgd-adp-ce_in-out']:
        version = loss_str
        attacks_to_run = [loss_str]
        adversary = AutoAttack(model, norm='Linf', verbose=False, eps=attack_eps, version=version,
                               attacks_to_run=attacks_to_run, num_in_classes=num_in_classes,
                               num_out_classes=num_out_classes, num_v_classes=num_v_classes, data_type='out')
        adversary.apgd.n_iter = attack_step
        adversary.apgd.n_restarts = 1
        adv_ood = adversary.run_standard_evaluation(x, y, bs=len(x), attack_all_emps=False, best_loss=best_loss)
        return adv_ood
    else:
        raise ValueError('un-supported loss_str: {}'.format(loss_str))


def pgd_attack_ood_misc(model, x, y, num_in_classes, num_out_classes=0, attack_step=10, attack_lr=0.003,
                    attack_eps=0.3, random_init=True, random_type='uniform', bn_type='eval', clamp=(0, 1),
                    loss_str='', best_loss=False):
    if attack_eps <= 0.0:
        return x
    if bn_type == 'eval':
        model.eval()
    elif bn_type == 'train':
        model.train()
    else:
        raise ValueError('error bn_type: {}'.format(bn_type))
    x_adv = x.clone().detach()
    if random_init:
        # Flag to use random initialization
        if random_type == 'gussian':
            x_adv = x_adv + 0.001 * torch.randn(x.shape, device=x.device)
        elif random_type == 'uniform':
            # x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * attack_eps
            random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-attack_eps, attack_eps).to(x_adv.device)
            x_adv = x_adv + random_noise
        else:
            raise ValueError('error random noise type: {}'.format(random_type))

    if best_loss:
        adv_best = x.detach().clone()
        loss_best = torch.ones([x.shape[0]]).to(x.device) * (-float('inf'))

    for i in range(attack_step):
        x_adv.requires_grad = True
        model.zero_grad()
        adv_logits = model(x_adv)
        if loss_str == 'pgd-ce':
            if len(y.size()) == 2:
                loss = nn_util.cross_entropy_soft_target(adv_logits, y, reduction='none')
            else:
                loss = F.cross_entropy(adv_logits, y, reduction='none')
        elif loss_str == 'pgd-ce_out':
            y_out = torch.max(adv_logits[:, num_in_classes:num_in_classes + num_out_classes], dim=1)[1] + num_in_classes
            loss = F.cross_entropy(adv_logits, y_out, reduction='none')
        elif loss_str == 'pgd-ce_v':
            num_real_classes = num_in_classes + num_out_classes
            assert adv_logits.size(1) > num_real_classes
            max_y_v = torch.max(adv_logits[:, num_real_classes:], dim=1)[1] + num_real_classes
            loss = F.cross_entropy(adv_logits, max_y_v, reduction='none')
        elif loss_str == 'pgd-ce_in':
            with torch.no_grad():
                _, preds = torch.max(adv_logits[:, :num_in_classes], dim=1)
            loss = -F.cross_entropy(adv_logits, preds, reduction='none')
        elif loss_str == 'pgd-ce_in-out':
            with torch.no_grad():
                _, preds = torch.max(adv_logits[:, :num_in_classes], dim=1)
            in_loss = -F.cross_entropy(adv_logits, preds, reduction='none')
            y_out = torch.max(adv_logits[:, num_in_classes:num_in_classes + num_out_classes], dim=1)[1] + num_in_classes
            out_loss = F.cross_entropy(adv_logits, y_out, reduction='none')
            loss = in_loss + out_loss
        elif loss_str == 'pgd-ce_rdmin-out':
            y_rdmin = torch.randint(0, num_in_classes, (len(adv_logits)), device=adv_logits.device)
            in_loss = -F.cross_entropy(adv_logits, y_rdmin, reduction='none')
            y_out = torch.max(adv_logits[:, num_in_classes:num_in_classes + num_out_classes], dim=1)[1] + num_in_classes
            out_loss = F.cross_entropy(adv_logits, y_out, reduction='none')
            loss = in_loss + out_loss
        elif loss_str == 'pgd-oe':
            y_unif = torch.zeros_like(adv_logits) + 1. / adv_logits.size(1)
            loss = nn_util.cross_entropy_soft_target(adv_logits, y_unif, reduction='none')
        elif loss_str == 'pgd-oe_in':
            y_unif = torch.zeros_like(adv_logits[:, :num_in_classes]) + 1. / adv_logits[:, :num_in_classes].size(1)
            loss = nn_util.cross_entropy_soft_target(adv_logits[:, :num_in_classes], y_unif, reduction='none')
        else:
            raise ValueError('un-supported loss: {}'.format(loss_str))

        loss.mean().backward()
        loss = loss.detach()
        grad = x_adv.grad.detach()
        grad = grad.sign()
        x_adv = x_adv.detach()
        x_adv = x_adv + attack_lr * grad

        # Projection
        x_adv = x + torch.clamp(x_adv - x, min=-attack_eps, max=attack_eps)
        x_adv = torch.clamp(x_adv, *clamp)

        if best_loss:
            ind_curr = loss > loss_best
            adv_best[ind_curr] = x_adv[ind_curr] + 0.
            loss_best[ind_curr] = loss[ind_curr] + 0.

    if best_loss:
        return adv_best
    return x_adv