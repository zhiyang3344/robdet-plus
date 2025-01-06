# Copyright (c) 2020-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
#

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from autoattack.other_utils import L0_norm, L1_norm, L2_norm



def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    #u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()
    
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)
    
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)
    
    s1 = -u.sum(dim=1)
    
    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)
    
    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    
    if c2.nelement != 0:
    
      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)
      
      #print(c2.shape, lb.shape)
      
      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
          
      while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2.)
        counter2 = counter4.type(torch.LongTensor)
        
        c8 = s[c2, counter2] + c[c2] < 0
        ind3 = c8.nonzero().squeeze(1)
        ind32 = (~c8).nonzero().squeeze(1)
        #print(ind3.shape)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]
        
        #print(lb, ub)
        counter += 1
        
      lb2 = lb.long()
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])
    
    return (sigma * d).view(x2.shape)





class APGDAttack():
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            num_in_classes=10,
            attack_other_in=False,
            num_out_classes=0,
            num_v_classes=0,
            data_type='in', ):
        """
        AutoPGD implementation in PyTorch
        """
        
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        #self.init_point = None
        self.use_largereps = use_largereps
        #self.larger_epss = None
        #self.iters = None
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.ce_st_dim = 0
        self.ce_end_dim = 10
        self.num_in_classes = num_in_classes
        self.attack_other_in = attack_other_in
        self.num_out_classes = num_out_classes
        self.num_v_classes = num_v_classes
        self.data_type = data_type
        assert self.data_type in ['in', 'out']
        assert self.num_in_classes >= 0
        assert self.num_out_classes >= 0
        assert self.num_v_classes >= 0


    def init_hyperparam(self, x):
        assert self.norm in ['Linf', 'L2', 'L1']
        assert not self.eps is None

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()


        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)
    
    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)


    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def ce_soft_target(self, logits, y_soft, reduction='mean'):
        assert len(y_soft.size()) == 2
        batch_size = logits.size()[0]
        log_prob = F.log_softmax(logits, dim=1)
        if reduction == 'none':
            loss = -torch.sum(log_prob * y_soft, dim=1)
        elif reduction == 'mean':
            loss = -torch.sum(log_prob * y_soft) / batch_size
        else:
            print('un-supported reduction: {}'.format(reduction))
        return loss

    def adp_oe_loss(self, logits, y):
        y_unif = torch.zeros_like(logits) + 1. / logits.size(1)
        # print('y_unif.size():', y_unif.size(), 'y_unif:', y_unif[:5])
        losses = self.ce_soft_target(logits, y_unif, reduction='none')
        return losses

    def adp_ce_in_loss(self, logits, y):
        return self.adaptive_ce_loss(logits, y, attack_in=True, attack_out=False, attack_v=False)

    def adp_ce_out_loss(self, logits, y):
        return self.adaptive_ce_loss(logits, y, attack_in=False, attack_out=True, attack_v=False)

    def adp_ce_v_loss(self, logits, y):
        return self.adaptive_ce_loss(logits, y, attack_in=False, attack_out=False, attack_v=True)

    def adp_ce_in_out_loss(self, logits, y):
        return self.adaptive_ce_loss(logits, y, attack_in=True, attack_out=True, attack_v=False)

    def adp_ce_in_v_loss(self, logits, y):
        return self.adaptive_ce_loss(logits, y, attack_in=True, attack_out=False, attack_v=True)

    def adp_ce_in_out_v_loss(self, logits, y):
        return self.adaptive_ce_loss(logits, y, attack_in=True, attack_out=True, attack_v=True)

    def get_pred_from_logits(self, logits, bs=256):
        pred = torch.zeros((logits.size(0),), device=logits.device).long()
        with torch.no_grad():
            n_batches = int(np.ceil(logits.shape[0] / bs))
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, logits.shape[0])
                batch_logits = logits[start_idx:end_idx, :]
                if self.data_type == 'in':
                    pred[start_idx:end_idx] = batch_logits[:, :self.num_in_classes].max(dim=1)[1]
                else:
                    if self.num_in_classes == logits.size(1) or (self.num_out_classes == 0 and self.num_v_classes == 0):
                        pred[start_idx:end_idx] = batch_logits[:, :self.num_in_classes].max(dim=1)[1]
                    elif self.num_out_classes > 0:
                        pred[start_idx:end_idx] = \
                        batch_logits[:, self.num_in_classes:self.num_in_classes + self.num_out_classes].max(dim=1)[
                            1] + self.num_in_classes
                    elif self.num_v_classes > 0:
                        pred[start_idx:end_idx] = \
                        batch_logits[:, self.num_in_classes:self.num_in_classes + self.num_v_classes].max(dim=1)[
                            1] + self.num_in_classes
        return pred


    def adaptive_ce_loss(self, logits, y, attack_in=True, attack_out=False, attack_v=F):
        assert attack_in | attack_out | attack_v == True
        num_in_classes = self.num_in_classes
        num_out_classes = self.num_out_classes
        num_v_classes = self.num_v_classes
        data_type = self.data_type
        target = self.y_target

        assert logits.size(1) == num_in_classes + num_out_classes + num_v_classes
        out_losses = 0
        if num_out_classes > 0 and attack_out:
            out_ind = logits[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[1] + num_in_classes
            out_losses = F.cross_entropy(logits, out_ind, reduction='none')
        v_losses = 0
        if num_v_classes > 0 and attack_v:
            v_ind = logits[:, num_in_classes + num_out_classes:].max(dim=1)[1] + num_in_classes + num_out_classes
            v_losses = F.cross_entropy(logits, v_ind, reduction='none')

        in_losses = 0
        if data_type == 'in':
            if attack_in:
                if target is not None:
                    in_losses = -F.cross_entropy(logits, target, reduction='none')
                else:
                    in_losses = F.cross_entropy(logits, y, reduction='none')
            losses = in_losses + out_losses + v_losses
        elif data_type == 'out':
            if attack_in:
                if target is not None:
                    in_losses = -F.cross_entropy(logits, target, reduction='none')
                else:
                    in_pred = logits[:, :num_in_classes].max(dim=1)[1]
                    in_losses = -F.cross_entropy(logits, in_pred, reduction='none')
            losses = in_losses + out_losses + v_losses
        else:
            raise ValueError('un-supported data_type'.format(data_type))

        return losses

    def cw_loss(self, logits, y):
        num_in_classes = self.num_in_classes
        data_type = self.data_type
        target = self.y_target

        u = torch.arange(logits.size(0))
        if data_type == 'in':
            if target is not None:
                losses = logits[u, target] - logits[u, y]
            else:
                with torch.no_grad():
                    temp_logits = logits.clone()
                    temp_logits[u, y] = -float('inf')
                    other_ind = temp_logits.max(dim=1)[1]
                losses = logits[u, other_ind] - logits[u, y]
        elif data_type == 'out':
            if target is not None:
                target_logit = logits[u, target]
                with torch.no_grad():
                    temp_logits = logits.clone()
                    temp_logits[u, target] = -float('inf')
                    other_ind = temp_logits.max(dim=1)[1]
                losses = target_logit - logits[u, other_ind]
            else:
                max_in_logits, max_in_ind = logits[:, :num_in_classes].max(dim=1)
                with torch.no_grad():
                    temp_logits = logits.clone()
                    temp_logits[u, max_in_ind] = -float('inf')
                    other_ind = temp_logits.max(dim=1)[1]
                losses = max_in_logits - logits[u, other_ind]
        else:
            raise ValueError('un-supported data_type'.format(data_type))
        return losses

    def adp_cw_in_loss(self, logits, y):
        return self.adaptive_cw_loss(logits, y, self.attack_other_in, attack_in=True, attack_out=False, attack_v=False)

    def adp_cw_v_loss(self, logits, y):
        return self.adaptive_cw_loss(logits, y, self.attack_other_in, attack_in=False, attack_out=False, attack_v=True)

    def adp_cw_out_loss(self, logits, y):
        return self.adaptive_cw_loss(logits, y, self.attack_other_in, attack_in=False, attack_out=True, attack_v=False)

    def adp_cw_in_out_loss(self, logits, y):
        return self.adaptive_cw_loss(logits, y, self.attack_other_in, attack_in=True, attack_out=True, attack_v=False)

    def adp_cw_in_v_loss(self, logits, y):
        return self.adaptive_cw_loss(logits, y, self.attack_other_in, attack_in=True, attack_out=False, attack_v=True)

    def adp_cw_in_out_v_loss(self, logits, y):
        return self.adaptive_cw_loss(logits, y, self.attack_other_in, attack_in=True, attack_out=True, attack_v=True)


    def adaptive_cw_loss(self, logits, y, attack_other_in, attack_in=True, attack_out=True, attack_v=False):
        num_in_classes = self.num_in_classes
        num_out_classes = self.num_out_classes
        num_v_classes = self.num_v_classes
        data_type = self.data_type
        target = self.y_target

        assert logits.size(1) == num_in_classes + num_out_classes + num_v_classes
        # num_v_classes = logits.size(1) - num_in_classes - num_out_classes
        max_out_logit = 0
        if num_out_classes > 0 and attack_out:
            max_out_logit = logits[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[0]

        max_v_logit = 0
        if num_v_classes > 0 and attack_v:
            max_v_logit = logits[:, num_in_classes + num_out_classes:].max(dim=1)[0]
        u = torch.arange(logits.size(0))
        if data_type == 'in':
            if target is not None:
                other_in_logit = logits[u, target]
            else:
                with torch.no_grad():
                    temp_logits = logits.clone()
                    temp_logits[u, y] = -float('inf')
                    in_max_ind = temp_logits[:, :num_in_classes].max(dim=1)[1]
                other_in_logit = logits[u, in_max_ind]
            corr_logit = logits[u, y]
            losses = other_in_logit - corr_logit - max_out_logit - max_v_logit
        elif data_type == 'out':
            if target is not None:
                target_in_logit = logits[u, target]
                other_in_logit = 0
                if attack_other_in:
                    with torch.no_grad():
                        temp_logits = logits.clone()
                        temp_logits[u, target] = -float('inf')
                        other_ind = temp_logits[:, :num_in_classes].max(dim=1)[1]
                    other_in_logit = logits[u, other_ind]
                losses = target_in_logit - other_in_logit - max_out_logit - max_v_logit
            else:
                assert (attack_in | attack_out | attack_v) == True
                top_logits, top_ind = logits[:, :num_in_classes].topk(k=2, dim=1)
                if attack_in:
                    other_in_logit = top_logits[:, 0]
                else:
                    other_in_logit = 0
                second_in_logit = 0
                if attack_other_in:
                    second_in_logit = top_logits[:, 1]
                losses = other_in_logit - second_in_logit - max_out_logit - max_v_logit
        else:
            raise ValueError('un-supported data_type'.format(data_type))

        return losses

    def adp_dlr_in_loss(self, logits, y):
        return self.adaptive_dlr_loss(logits, y, self.attack_other_in, attack_out=False, attack_v=False)

    def adp_dlr_in_out_loss(self, logits, y):
        return self.adaptive_dlr_loss(logits, y, self.attack_other_in, attack_out=True, attack_v=False)

    def adp_dlr_in_v_loss(self, logits, y):
        return self.adaptive_dlr_loss(logits, y, self.attack_other_in, attack_out=False, attack_v=True)

    def adp_dlr_in_out_v_loss(self, logits, y):
        return self.adaptive_dlr_loss(logits, y, self.attack_other_in, attack_out=True, attack_v=True)

    def adaptive_dlr_loss(self, logits, y, attack_other_in, attack_out, attack_v):
        num_in_classes = self.num_in_classes
        num_out_classes = self.num_out_classes
        num_v_classes = self.num_v_classes
        data_type = self.data_type
        target = self.y_target

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
        u = torch.arange(logits.size(0))
        if data_type == 'in':
            if target is not None:
                other_in_logit=logits[u, target]
            else:
                with torch.no_grad():
                    temp_logits = logits.clone()
                    temp_logits[u, y] = -float('inf')
                    other_in_ind = temp_logits[:, :num_in_classes].max(dim=1)[1]
                other_in_logit = logits[u, other_in_ind]
            corr_logit = logits[u, y]
            losses = (other_in_logit - corr_logit - max_out_logit - max_v_logit) / (
                        whole_in_max - whole_in_3th + 1e-12)
        elif data_type == 'out':
            if target is not  None:
                target_in_logit = logits[u, target]
                other_in_logit = 0
                if attack_other_in:
                    with torch.no_grad():
                        temp_logits = logits.clone()
                        temp_logits[u, target] = -float('inf')
                        other_in_ind = temp_logits[:, :num_in_classes].max(dim=1)[1]
                    other_in_logit = logits[u, other_in_ind]
                losses = (target_in_logit - other_in_logit - max_out_logit - max_v_logit) / (
                        whole_in_max - whole_in_3th + 1e-12)
            else:
                top_logits, top_ind = logits[:, :num_in_classes].topk(k=2, dim=1)
                max_in_logit = top_logits[:, 0]
                second_in_logit = 0
                if attack_other_in:
                    second_in_logit = top_logits[:, 1]
                losses = (max_in_logit - second_in_logit - max_out_logit - max_v_logit) / (
                        whole_in_max - whole_in_3th + 1e-12)
        else:
            raise ValueError('un-supported data_type'.format(data_type))

        return losses

    def dlr_loss(self, logits, y):
        target = self.y_target
        data_type = self.data_type
        num_in_classes = self.num_in_classes

        logits_sorted, ind_sorted = logits[:, :num_in_classes].sort(dim=1)
        u = torch.arange(logits.shape[0])
        if data_type == 'in':
            if target is not None:
                target_logit = logits[u, target]
                losses = (target_logit - logits[u, y]) / (logits_sorted[:, -1] - logits_sorted[:, -3] + 1e-12)
            else:
                ind = (ind_sorted[:, -1] == y).float()
                losses = -(logits[u, y] - logits_sorted[:, -2] * ind - logits_sorted[:, -1] * (1. - ind)) / \
                         (logits_sorted[:, -1] - logits_sorted[:, -3] + 1e-12)
        elif data_type == 'out':
            if target is not None:
                target_logit = logits[u, target]
                with torch.no_grad():
                    temp_logits = logits.clone()
                    temp_logits[u, target] = -float('inf')
                    other_ind = temp_logits.max(dim=1)[1]
                other_logit = logits[u, other_ind]
                losses = (target_logit - other_logit) / (logits_sorted[:, -1] - logits_sorted[:, -3] + 1e-12)
            else:
                max_in_logit, max_in_ind = logits[:, :num_in_classes].max(dim=1)
                with torch.no_grad():
                    temp_logits = logits.clone()
                    temp_logits[u, max_in_ind] = -float('inf')
                    other_ind = temp_logits.max(dim=1)[1]
                other_logit = logits[u, other_ind]
                losses = (max_in_logit - other_logit) / (logits_sorted[:, -1] - logits_sorted[:, -3] + 1e-12)
        else:
            raise ValueError('un-supported data_type {}'.format(data_type))
        return losses


    def get_acc_flags(self, logits, y):
        pred = self.get_pred_from_logits(logits, bs=logits.shape[0])
        if self.data_type == 'in':
            acc = pred == y
        else:
            acc = y == y
            # if self.num_in_classes == logits.size(1) or (self.num_out_classes == 0 and self.num_v_classes == 0):
            #     ## the OOD label is a uniform distribution
            #     u = torch.arange(0, logits.shape[0])
            #     with torch.no_grad():
            #         temp_logits = logits.clone()
            #         temp_logits[u, y] = float('-inf')
            #         top_logits = temp_logits.topk(k=2, dim=1)[0]
            #         acc = top_logits[:, 0] < top_logits[:, 1] + logits[u, y]
            # elif self.num_out_classes > 0:
            #     max_in = logits[:, :self.num_in_classes].max(dim=1)[0]
            #     sum_out = torch.sum(logits[:, self.num_in_classes:self.num_in_classes + self.num_out_classes], dim=1)
            #     acc = sum_out > max_in
            # elif self.num_v_classes > 0:
            #     ## the virtual classes is a uniform distribution
            #     with torch.no_grad():
            #         top_logits = logits[:, :self.num_in_classes].topk(k=2, dim=1)[0]
            #         max_v = logits[:, self.num_in_classes:self.num_in_classes + self.num_v_classes].max(dim=1)[0]
            #         acc = top_logits[:, 0] < (top_logits[:, 1] + max_v)
            # else:
            #     raise ValueError('plz check self.num_in_classes:{}, self.num_out_classes:{}, '
            #                      'self.num_v_classes:{}'.format(self.num_in_classes, self.num_out_classes,
            #                                                     self.num_v_classes))
        return acc


    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta

        if not x_init is None:
            x_adv = x_init.clone()
            if self.norm == 'L1' and self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
            ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
            ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        if not self.is_tf_model:
            if self.loss == 'ce':
                criterion_indiv = nn.CrossEntropyLoss(reduction='none')
            elif self.loss == 'oe':
                criterion_indiv = self.adp_oe_loss
            elif self.loss == 'adp-ce_in':
                criterion_indiv = self.adp_ce_in_loss
            elif self.loss == 'adp-ce_out':
                criterion_indiv = self.adp_ce_out_loss
            elif self.loss == 'adp-ce_v':
                criterion_indiv = self.adp_ce_v_loss
            elif self.loss == 'adp-ce_in-out':
                criterion_indiv = self.adp_ce_in_out_loss
            elif self.loss == 'adp-ce_in-v':
                criterion_indiv = self.adp_ce_in_v_loss
            elif self.loss == 'adp-ce_in-out-v':
                criterion_indiv = self.adp_ce_in_out_v_loss

            elif self.loss == 'cw':
                criterion_indiv = self.cw_loss
            elif self.loss == 'adp-cw_in':
                criterion_indiv = self.adp_cw_in_loss
            elif self.loss == 'adp-cw_v':
                criterion_indiv = self.adp_cw_v_loss
            elif self.loss == 'adp-cw_out':
                criterion_indiv = self.adp_cw_out_loss
            elif self.loss == 'adp-cw_in-out':
                criterion_indiv = self.adp_cw_in_out_loss
            elif self.loss == 'adp-cw_in-v':
                criterion_indiv = self.adp_cw_in_v_loss
            elif self.loss == 'adp-cw_in-out-v':
                criterion_indiv = self.adp_cw_in_out_v_loss

            elif self.loss == 'dlr':
                criterion_indiv = self.dlr_loss
            elif self.loss == 'adp-dlr_in':
                criterion_indiv = self.adp_dlr_in_loss
            elif self.loss == 'adp-dlr_in-out':
                criterion_indiv = self.adp_dlr_in_out_loss
            elif self.loss == 'adp-dlr_in-v':
                criterion_indiv = self.adp_dlr_in_v_loss
            elif self.loss == 'adp-dlr_in-out-v':
                criterion_indiv = self.adp_dlr_in_out_v_loss
            elif self.loss == 'ce-targeted-cfts':
                criterion_indiv = lambda x, y: -1. * F.cross_entropy(x, y,
                    reduction='none')

            elif self.loss == 'ce-targeted':
                criterion_indiv = self.ce_loss_targeted

            # elif self.loss == 'cw-targeted':
            #     criterion_indiv = self.cw_loss_targeted
            # elif self.loss == 'adp-cw_in-targeted':
            #     criterion_indiv = self.adp_cw_in_loss_targeted
            # elif self.loss == 'adp-cw_in-out-targeted':
            #     criterion_indiv = self.adp_cw_in_out_loss_targeted
            # elif self.loss == 'adp-cw_in-v-targeted':
            #     criterion_indiv = self.adp_cw_in_v_loss_targeted

            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.dlr_loss_targeted
            # elif self.loss == 'adp-dlr_in-targeted':
            #     criterion_indiv = self.adp_dlr_in_loss_targeted
            # elif self.loss == 'adp-dlr_in-out-targeted':
            #     criterion_indiv = self.adp_dlr_in_out_loss_targeted
            # elif self.loss == 'adp-dlr_in-v-targeted':
            #     criterion_indiv = self.adp_dlr_in_v_loss_targeted
            else:
                raise ValueError('unknowkn loss')
        else:
            if self.loss == 'ce':
                criterion_indiv = self.model.get_logits_loss_grad_xent
            elif self.loss == 'dlr':
                criterion_indiv = self.model.get_logits_loss_grad_dlr
            elif self.loss == 'dlr-targeted':
                criterion_indiv = self.model.get_logits_loss_grad_target
            else:
                raise ValueError('unknowkn loss')
        
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            if not self.is_tf_model:
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                    # print('---------------11111---------------->loss_indiv', loss_indiv[:5])
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            else:
                if self.y_target is None:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                else:
                    logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y,
                        self.y_target)
                grad += grad_curr
        
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc=self.get_acc_flags(logits, y)
        # print('acc flags.size():', acc.size(), 'sum:', acc.sum())
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.ndims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        if self.norm == 'L1':
            k = max(int(.04 * self.n_iter), 1)
            n_fts = math.prod(self.orig_dim)
            if x_init is None:
                topk = .2 * torch.ones([x.shape[0]], device=self.device)
                sp_old =  n_fts * torch.ones_like(topk)
            else:
                topk = L0_norm(x_adv - x) / n_fts / 1.5
                sp_old = L0_norm(x_adv - x)
            #print(topk[0], sp_old[0])
            adasp_redstep = 1.5
            adasp_minstep = 10.
            #print(step_size[0].item())
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]        
        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                        x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        self.lp_norm(x_adv_1 - x)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        self.lp_norm(x_adv_1 - x)), 0.0, 1.0)

                elif self.norm == 'L1':
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                    grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                        sparsegrad.sign().abs().view(x.shape[0], -1).sum(dim=-1).view(
                        -1, *[1]*(len(x.shape) - 1)) + 1e-10)
                    
                    delta_u = x_adv_1 - x
                    delta_p = L1_projection(x, delta_u, self.eps)
                    x_adv_1 = x + delta_u + delta_p

                x_adv = x_adv_1 + 0.

            ## get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                if not self.is_tf_model:
                    with torch.enable_grad():
                        logits = self.model(x_adv)
                        loss_indiv = criterion_indiv(logits, y)
                        loss = loss_indiv.sum()
                        # print('---------------22222---------------->loss_indiv', loss_indiv[:5])
                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                else:
                    if self.y_target is None:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y)
                    else:
                        logits, loss_indiv, grad_curr = criterion_indiv(x_adv, y, self.y_target)
                    grad += grad_curr
            
            grad /= float(self.eot_iter)

            acc_cur = self.get_acc_flags(logits, y)
            # print('acc_cur flags.size():', acc.size(), 'sum:', acc.sum())
            acc = torch.min(acc, acc_cur)
            acc_steps[i + 1] = acc + 0
            ind_pred = (acc_cur == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), acc.float().mean(), str_stats))
                #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1 + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0

              counter3 += 1

              if counter3 == k:
                  if self.norm in ['Linf', 'L2']:
                      fl_oscillation = self.check_oscillation(loss_steps, i, k,
                          loss_best, k3=self.thr_decr)
                      fl_reduce_no_impr = (1. - reduced_last_check) * (
                          loss_best_last_check >= loss_best).float()
                      fl_oscillation = torch.max(fl_oscillation,
                          fl_reduce_no_impr)
                      reduced_last_check = fl_oscillation.clone()
                      loss_best_last_check = loss_best.clone()
    
                      if fl_oscillation.sum() > 0:
                          ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                          step_size[ind_fl_osc] /= 2.0
                          n_reduced = fl_oscillation.sum()
    
                          x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                          grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                      k = max(k - self.size_decr, self.n_iter_min)
                  
                  elif self.norm == 'L1':
                      sp_curr = L0_norm(x_best - x)
                      fl_redtopk = (sp_curr / sp_old) < .95
                      topk = sp_curr / n_fts / 1.5
                      step_size[fl_redtopk] = alpha * self.eps
                      step_size[~fl_redtopk] /= adasp_redstep
                      step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)
                      sp_old = sp_curr.clone()
                  
                      x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                      grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                  
                  counter3 = 0
                  #k = max(k - self.size_decr, self.n_iter_min)

        #
        
        return (x_best, acc, loss_best, x_best_adv)

    def perturb(self, x, y=None, attack_all_emps=False, best_loss=False, x_init=None, target=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """

        assert self.loss in ['ce', 'adp-ce_in', 'adp-ce_out', 'adp-ce_v', 'adp-ce_in-out', 'adp-ce_in-v', 'adp-ce_in-out-v',
                             'cw', 'adp-cw_in', 'adp-cw_out', 'adp-cw_v', 'adp-cw_in-out', 'adp-cw_in-v', 'adp-cw_in-out-v',
                             'dlr', 'adp-dlr_in', 'adp-dlr_in-out', 'adp-dlr_in-v', 'adp-dlr_in-out-v',
                             'oe', ]  # 'ce-targeted-cfts'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            logits = self.model(x)
        else:
            logits = self.model.predict(x)
        # create un-targeted label
        y_pred = self.get_pred_from_logits(logits, bs=x.shape[0])
        if y is None:
            #y_pred = self.predict(x).max(1)[1]
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        # print('y', y)
        adv = x.clone()
        acc = self.get_acc_flags(logits, y)
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        if self.use_largereps:
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1]) # make sure to use the given iterations
            if self.verbose:
                print('using schedule [{}x{}]'.format('+'.join([str(c
                    ) for c in epss]), '+'.join([str(c) for c in iters])))
        
        startt = time.time()
        # if not best_loss:
        if not attack_all_emps:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)
            if best_loss:
                adv_best = x.detach().clone()
                loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    if target is not None:
                        self.y_target = target[ind_to_fool].clone()
                        # with torch.no_grad():
                        #     _, top_ind = temp_output[:, :self.num_in_classes][ind_to_fool].topk(k=2, dim=1)
                        #     print('y_to_fool', y_to_fool)
                        #     print('top_ind1', top_ind[:, 0])
                        #     print('target', self.y_target)
                        #     print('top_ind2', top_ind[:, 1])
                        #     print()
                        #     exit()
                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr
                    ind_curr = (acc_curr == 0).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))
                    if best_loss:
                        ind_satisfied = (loss_curr > loss_best[ind_to_fool]).nonzero().squeeze()
                        adv_best[ind_to_fool[ind_satisfied]] = best_curr[ind_satisfied] + 0.
                        loss_best[ind_to_fool[ind_satisfied]] = loss_curr[ind_satisfied] + 0.
            if best_loss:
                return adv_best
            return adv
        else:
            if target is not None:
                self.y_target = target.clone()
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.
                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
        eps_target = float(epss[-1])
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            #
            if not x_init is None:
                x_init += L1_projection(x, x_init - x, 1. * eps)
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)

        return (x_init, acc, loss, x_adv)


class APGDAttack_targeted(APGDAttack):
    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=None,
            seed=0,
            eot_iter=1,
            rho=.75,
            topk=None,
            n_target_classes=9,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            num_in_classes=10,
            num_out_classes=0,
            num_v_classes=0,
            data_type='in'
            ):
        """
        AutoPGD on the targeted DLR loss
        """
        super(APGDAttack_targeted, self).__init__(predict, n_iter=n_iter, norm=norm,
            n_restarts=n_restarts, eps=eps, seed=seed, loss='dlr-targeted',
            eot_iter=eot_iter, rho=rho, topk=topk, verbose=verbose, device=device,
            use_largereps=use_largereps, is_tf_model=is_tf_model)

        self.y_target = None
        self.n_target_classes = n_target_classes
        self.num_in_classes = num_in_classes
        self.num_out_classes = num_out_classes
        self.num_v_classes = num_v_classes
        self.data_type = data_type
        assert self.data_type in ['in', 'out']
        assert self.num_in_classes >= 0
        assert self.num_out_classes >= 0
        assert self.num_v_classes >= 0


    def dlr_loss_targeted(self, logits, y):
        u = torch.arange(logits.shape[0])
        logits_sorted, _ = logits.sort(dim=1)
        if self.data_type == 'in':
            return -(logits[u, y] - logits[u, self.y_target]) / (
                        logits_sorted[:, -1] - .5 * (logits_sorted[:, -3] + logits_sorted[:, -4]) + 1e-12)
        elif self.data_type == 'out':
            with torch.no_grad():
                temp_logits = logits.clone()
                temp_logits[u, self.y_target] = -float('inf')
                other_ind = temp_logits.max(dim=1)[1]
            other_logit = logits[u, other_ind]
            return (logits[u, self.y_target] - other_logit) / (
                    logits_sorted[:, -1] - .5 * (logits_sorted[:, -3] + logits_sorted[:, -4]) + 1e-12)
        else:
            raise ValueError('un-supported data_type: {}'.format(self.data_type))


    def cw_loss_targeted(self, logits, y):
        u = torch.arange(logits.shape[0])
        if self.data_type == 'in':
            return logits[u, self.y_target] - logits[u, y]
        elif self.data_type == 'out':
            with torch.no_grad():
                temp_logits = logits.clone()
                temp_logits[u, self.y_target] = -float('inf')
                other_ind = temp_logits.max(dim=1)[1]
            other_logit = logits[u, other_ind]
            return logits[u, self.y_target] - other_logit
        else:
            raise ValueError('un-supported data_type: {}'.format(self.data_type))

    def adp_dlr_in_loss_targeted(self, logits, y):
        return self.adaptive_dlr_loss_targeted(logits, y, self.y_target, self.attack_other_in, attack_out=False,
                                               attack_v=False)

    def adp_dlr_in_out_loss_targeted(self, logits, y):
        return self.adaptive_dlr_loss_targeted(logits, y, self.y_target, self.attack_other_in, attack_out=True,
                                               attack_v=False)

    def adp_dlr_in_v_loss_targeted(self, logits, y):
        return self.adaptive_dlr_loss_targeted(logits, y, self.y_target, self.attack_other_in, attack_out=False,
                                               attack_v=True)

    def adaptive_dlr_loss_targeted(self, logits, y, y_target, attack_other_in, attack_out, attack_v):
        num_in_classes = self.num_in_classes
        num_out_classes = self.num_out_classes
        num_v_classes = self.num_v_classes
        data_type = self.data_type

        assert logits.size(1) == num_in_classes + num_out_classes + num_v_classes
        assert y_target < num_in_classes
        u = torch.arange(logits.shape[0])
        if data_type=='in':
            logits_sorted, _ = logits.sort(dim=1)
            return -(logits[u, y] - logits[u, y_target]) / (
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
            return (logits[u, y_target] - other_in_logit - max_out_logit - max_v_logit) / (
                    logits_sorted[:, -1] - .5 * (logits_sorted[:, -3] + logits_sorted[:, -4]) + 1e-12)
        else:
            raise  ValueError('un-supported data_type: {}'.format(data_type))

    def adp_cw_in_loss_targeted(self, logits, y):
        return self.adaptive_cw_loss_targeted(logits, y, self.y_target, self.attack_other_in, attack_out=False,
                                              attack_v=False)

    def adp_cw_in_out_loss_targeted(self, logits, y):
        return self.adaptive_cw_loss_targeted(logits, y, self.y_target, self.attack_other_in, attack_out=True,
                                              attack_v=False)

    def adp_cw_in_v_loss_targeted(self, logits, y):
        return self.adaptive_cw_loss_targeted(logits, y, self.y_target, self.attack_other_in, attack_out=False,
                                              attack_v=True)

    def adaptive_cw_loss_targeted(self, logits, y, y_target, attack_other_in, attack_out, attack_v):
        num_in_classes = self.num_in_classes
        num_out_classes = self.num_out_classes
        num_v_classes = self.num_v_classes
        data_type = self.data_type

        assert logits.size(1) == num_in_classes + num_out_classes + num_v_classes
        assert y_target < num_in_classes
        u = torch.arange(logits.shape[0])
        if data_type == 'in':
            logits_sorted, _ = logits.sort(dim=1)
            return logits[u, y_target] - logits[u, y]
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

            return logits[u, y_target] - other_in_logit - max_out_logit - max_v_logit
        else:
            raise ValueError('un-supported data_type: {}'.format(data_type))


    def ce_loss_targeted(self, x, y):
        return -1. * F.cross_entropy(x, self.y_target, reduction='none')

    def perturb(self, x, y=None, attack_all_emps=False, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        """

        assert self.loss in ['ce-targeted', 'dlr-targeted']
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        if not self.is_tf_model:
            temp_output = self.model(x)
        else:
            temp_output = self.model.predict(x)

        # create target label
        y_pred = self.get_pred_from_logits(temp_output, bs=x.shape[0])
        if y is None:
            # y_pred = self._get_predicted_label(x)
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)

        adv = x.clone()
        acc = y_pred == y

        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        startt = time.time()

        if not attack_all_emps:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)
            if best_loss:
                adv_best = x.detach().clone()
                loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
            if self.use_largereps:
                epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
                iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
                    .4 * self.n_iter_orig]
                iters = [math.ceil(c) for c in iters]
                iters[-1] = self.n_iter_orig - sum(iters[:-1])
                if self.verbose:
                    print('using schedule [{}x{}]'.format('+'.join([str(c
                        ) for c in epss]), '+'.join([str(c) for c in iters])))

            if self.data_type == 'in':
                start_target_class = 2
            else:
                start_target_class = 1
            end_target_class = start_target_class + self.n_target_classes
            for target_class in range(start_target_class, end_target_class):
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool = x[ind_to_fool].clone()
                        y_to_fool = y[ind_to_fool].clone()

                        if not self.is_tf_model:
                            output = self.model(x_to_fool)
                        else:
                            output = self.model.predict(x_to_fool)
                        if self.data_type == 'in':
                            self.y_target = output.sort(dim=1)[1][:, -target_class]
                        else:
                            self.y_target = output[:, :self.num_in_classes].sort(dim=1)[1][:, -target_class]
                        # print('self.y_target:{}'.format(self.y_target))

                        if not self.use_largereps:
                            res_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        else:
                            res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                        best_curr, acc_curr, loss_curr, adv_curr = res_curr
                        ind_curr = (acc_curr == 0).nonzero().squeeze()

                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print('target class {}'.format(target_class),
                                '- restart {} - robust accuracy: {:.2%}'.format(
                                counter, acc.float().mean()),
                                '- cum. time: {:.1f} s'.format(
                                time.time() - startt))
                        if best_loss:
                            ind_satisfied = (loss_curr > loss_best[ind_to_fool]).nonzero().squeeze()
                            adv_best[ind_to_fool[ind_satisfied]] = best_curr[ind_satisfied] + 0.
                            loss_best[ind_to_fool[ind_satisfied]] = loss_curr[ind_satisfied] + 0.
            if best_loss:
                return adv_best
            return adv
        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
            if self.data_type == 'in':
                start_target_class = 2
            else:
                start_target_class = 1
            end_target_class = start_target_class + self.n_target_classes
            for target_class in range(start_target_class, end_target_class):
                for counter in range(self.n_restarts):
                    if not self.is_tf_model:
                        output = self.model(x)
                    else:
                        output = self.model.predict(x)
                    if self.data_type == 'in':
                        self.y_target = output.sort(dim=1)[1][:, -target_class]
                    else:
                        self.y_target = output[:, :self.num_in_classes].sort(dim=1)[1][:, -target_class]
                    # print('self.y_target:{}'.format(self.y_target))
                    best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                    ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                    adv_best[ind_curr] = best_curr[ind_curr] + 0.
                    # print('loss_best[ind_curr]', loss_best[ind_curr][:5])
                    loss_best[ind_curr] = loss_curr[ind_curr] + 0.
                    # print('loss_curr[ind_curr]', loss_curr[ind_curr][:5])
                    if self.verbose:
                        print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
                    print()
            return adv_best

