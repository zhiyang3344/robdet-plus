import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse
import time
import math
from .other_utils import Logger


class AutoAttack():
    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True, attacks_to_run=[], version='standard',
                 is_tf_model=False, device='cuda', log_path=None, num_in_classes=10, num_out_classes=0, num_v_classes=0,
                 data_type='in'):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = Logger(log_path)
        self.num_in_classes = num_in_classes
        self.num_out_classes = num_out_classes
        self.num_v_classes = num_v_classes
        self.data_type = data_type
        assert self.data_type in ['in', 'out']
        assert self.num_in_classes >= 0
        assert self.num_out_classes >= 0
        assert self.num_v_classes >= 0

        if not self.is_tf_model:
            from .autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False, eps=self.epsilon,
                                   norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                                   num_in_classes=num_in_classes, num_out_classes=num_out_classes,
                                   num_v_classes=num_v_classes, data_type=data_type)

            from .fab_pt import FABAttack_PT
            self.fab = FABAttack_PT(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                                    norm=self.norm, verbose=False, device=self.device, num_in_classes=num_in_classes)

            from .square import SquareAttack
            self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                                       n_restarts=1, seed=self.seed, verbose=False, device=self.device,
                                       resc_schedule=False, num_in_classes=num_in_classes)

            from .autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
                                                     eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75,
                                                     seed=self.seed, device=self.device,
                                                     num_in_classes=self.num_in_classes,
                                                     num_out_classes=num_out_classes, num_v_classes=num_v_classes,
                                                     data_type=data_type)

        else:
            from .autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False, eps=self.epsilon,
                                   norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                                   is_tf_model=True, num_in_classes=num_in_classes, num_out_classes=num_out_classes,
                                   num_v_classes=num_v_classes, data_type=data_type)

            from .fab_tf import FABAttack_TF
            self.fab = FABAttack_TF(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                                    norm=self.norm, verbose=False, device=self.device, num_in_classes=num_in_classes)

            from .square import SquareAttack
            self.square = SquareAttack(self.model.predict, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                                       n_restarts=1, seed=self.seed, verbose=False, device=self.device,
                                       resc_schedule=False, num_in_classes=num_in_classes)

            from .autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
                                                     eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75,
                                                     seed=self.seed, device=self.device, is_tf_model=True,
                                                     num_in_classes=num_in_classes, num_out_classes=num_out_classes,
                                                     num_v_classes=num_v_classes, data_type=data_type)
    
        if version in ['standard', 'plus', 'rand']:
            self.set_version(version)
        
    def get_logits(self, x):
        if not self.is_tf_model:
            return self.model(x)
        else:
            return self.model.predict(x)
    
    def get_seed(self):
        return time.time() if self.seed is None else self.seed

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


    def get_pred_label(self, x_orig, bs=250):
        with torch.no_grad():
            y_org = torch.zeros((x_orig.size(0),), device=x_orig.device).long()
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])
                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                output = self.get_logits(x)
                if self.data_type == 'in':
                    y_org[start_idx:end_idx] = output[:, :self.num_in_classes].max(dim=1)[1]
                else:
                    if self.num_in_classes == output.size(1) or (self.num_out_classes == 0 and self.num_v_classes == 0):
                        y_org[start_idx:end_idx] = output[:, :self.num_in_classes].max(dim=1)[1]
                    elif self.num_out_classes > 0:
                        y_org[start_idx:end_idx] = \
                            output[:, self.num_in_classes:self.num_in_classes + self.num_out_classes].max(dim=1)[
                                1] + self.num_in_classes
                    elif self.num_v_classes > 0:
                        y_org[start_idx:end_idx] = \
                            output[:, self.num_in_classes:self.num_in_classes + self.num_v_classes].max(dim=1)[
                                1] + self.num_in_classes
                    else:
                        raise ValueError('plz check self.num_in_classes:{}, self.num_out_classes:{}, '
                                         'self.num_v_classes:{}'.format(self.num_in_classes, self.num_out_classes,
                                                                        self.num_v_classes))
        return y_org

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


    def run_standard_evaluation(self, x_orig, y_orig, bs=250, attack_all_emps=False, best_loss=False, target=None):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        if y_orig is None:
            if self.verbose:
                print('passed y is None, I will use the pred as y')
            y_orig = self.get_pred_label(x_orig, bs)

        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)

            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])
                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                correct_batch = self.get_acc_flags(self.get_logits(x), y)
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)
                if target is not None:
                    y_target = target[start_idx:end_idx].clone().to(self.device)

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
            if self.verbose:
                self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))

            x_adv = x_orig.clone().detach()
            startt = time.time()
            for attack in self.attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                if self.data_type == 'in' and (not attack_all_emps):
                    num_robust = torch.sum(robust_flags).item()
                    robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                else:
                    num_robust = x_orig.shape[0]
                    robust_lin_idcs = torch.nonzero(y_orig == y_orig, as_tuple=False)
                if num_robust == 0:
                    break
                n_batches = int(np.ceil(num_robust / bs))

                if num_robust > 1:
                    robust_lin_idcs.squeeze_()
                # num_newly_miscls_to_v = 0
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)
                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    # print('batch_datapoint_idcs:', batch_datapoint_idcs)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)
                    y_target = None
                    if target is not None:
                        y_target = target[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)

                    # nat_logits = self.get_logits(x)
                    # _, nat_whole_pred = torch.max(nat_logits, dim=1)
                    # nat_pred_vidcs = nat_whole_pred >= self.num_in_classes + self.num_out_classes
                    # nat_pred_in_idcs = nat_whole_pred < self.num_in_classes
                    # print('num of nat_pred_in_virtual_idcs:', nat_pred_in_virtual_idcs.sum().item())
                    # print('nat_pred_in_virtual:{}'.format(nat_logits[nat_pred_in_virtual_idcs]))

                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)

                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss) #cheap=True

                    elif attack == 'fab':
                        # fab
                        self.fab.targeted = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)

                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss) #cheap=True

                    elif attack == 'fab-t':
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)

                    elif attack == 'apgd-oe':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'oe'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)

                    elif attack == 'apgd-adp-ce_in':
                        self.apgd.loss = 'adp-ce_in'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-ce_out':
                        self.apgd.loss = 'adp-ce_out'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-ce_v':
                        self.apgd.loss = 'adp-ce_v'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-ce_in-out':
                        self.apgd.loss = 'adp-ce_in-out'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-ce_in-v':
                        self.apgd.loss = 'adp-ce_in-v'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-ce_in-out-v':
                        self.apgd.loss = 'adp-ce_in-out-v'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss)  # cheap=True

                    elif attack == 'apgd-cw':
                        self.apgd.loss = 'cw'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-cw_in':
                        self.apgd.loss = 'adp-cw_in'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-cw_out':
                        self.apgd.loss = 'adp-cw_out'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-cw_v':
                        self.apgd.loss = 'adp-cw_v'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-cw_in-out':
                        self.apgd.loss = 'adp-cw_in-out'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-cw_in-v':
                        self.apgd.loss = 'adp-cw_in-v'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-cw_in-out-v':
                        self.apgd.loss = 'adp-cw_in-out-v'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True

                    elif attack == 'apgd-adp-dlr_in':
                        self.apgd.loss = 'adp-dlr_in'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-dlr_in-out':
                        self.apgd.loss = 'adp-dlr_in-out'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-dlr_in-v':
                        self.apgd.loss = 'adp-dlr_in-v'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-dlr_in-out-v':
                        self.apgd.loss = 'adp-dlr_in-out-v'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True

                    # elif attack == 'adaptive-square':
                    #     self.square.loss = 'adaptive-margin'
                    #     self.square.seed = self.get_seed()
                    #     adv_curr = self.square.perturb(x, y)
                    #
                    elif attack == 'apgd-ce-targeted':
                        # targeted apgd
                        self.apgd.loss = 'ce-targeted'
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y, attack_all_emps=attack_all_emps,
                                                              best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-ce_in-targeted':
                        # targeted apgd
                        self.apgd.loss = 'adp-ce_in'
                        self.apgd.seed = self.get_seed()
                        assert target is not None
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss, target=y_target)  # cheap=True
                    elif attack == 'apgd-adp-ce_in-out-targeted':
                        # targeted apgd
                        self.apgd.loss = 'adp-ce_in-out'
                        self.apgd.seed = self.get_seed()
                        assert target is not None
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss, target=y_target)  # cheap=True
                    elif attack == 'apgd-adp-ce_in-v-targeted':
                        # targeted apgd
                        self.apgd.loss = 'adp-ce_in-v'
                        self.apgd.seed = self.get_seed()
                        assert target is not None
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss, target=y_target)  # cheap=True
                    elif attack == 'apgd-adp-ce_in-out-v-targeted':
                        # targeted apgd
                        self.apgd.loss = 'adp-ce_in-out-v'
                        self.apgd.seed = self.get_seed()
                        assert target is not None
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss, target=y_target)  # cheap=True

                    elif attack == 'apgd-cw-targeted':
                        # targeted apgd
                        self.apgd.loss = 'cw-targeted'
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y, attack_all_emps=attack_all_emps,
                                                              best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-cw-targeted':
                        self.apgd.loss = 'cw'
                        self.apgd.seed = self.get_seed()
                        assert target is not None
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss, target=y_target)  # cheap=True

                    elif attack == 'apgd-adp-cw_in-targeted':
                        self.apgd.loss = 'adp-cw_in'
                        self.apgd.seed = self.get_seed()
                        assert target is not None
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss, target=y_target)  # cheap=True
                    elif attack == 'apgd-adp-cw_in-out-targeted':
                        self.apgd.loss = 'adp-cw_in-out'
                        self.apgd.seed = self.get_seed()
                        assert target is not None
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss, target=y_target)  # cheap=True
                    elif attack == 'apgd-adp-cw_in-v-targeted':
                        self.apgd.loss = 'adp-cw_in-v'
                        self.apgd.seed = self.get_seed()
                        assert target is not None
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss, target=y_target)  # cheap=True
                    elif attack == 'apgd-adp-cw_in-out-v-targeted':
                        self.apgd.loss = 'adp-cw_in-out-v'
                        self.apgd.seed = self.get_seed()
                        assert target is not None
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss, target=y_target)  # cheap=True

                    # elif attack == 'apgd-adp-cw_in-targeted':
                    #     # targeted apgd
                    #     self.apgd.loss = 'adp-cw_in-targeted'
                    #     self.apgd_targeted.seed = self.get_seed()
                    #     adv_curr = self.apgd_targeted.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss) #cheap=True
                    # elif attack == 'apgd-adp-cw_in-out-targeted':
                    #     # targeted apgd
                    #     self.apgd.loss = 'adp-cw_in-out-targeted'
                    #     self.apgd_targeted.seed = self.get_seed()
                    #     adv_curr = self.apgd_targeted.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    # elif attack == 'apgd-adp-cw_in-v-targeted':
                    #     # targeted apgd
                    #     self.apgd.loss = 'adp-cw_in-v-targeted'
                    #     self.apgd_targeted.seed = self.get_seed()
                    #     adv_curr = self.apgd_targeted.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True

                    elif attack == 'apgd-dlr-targeted':
                        # targeted apgd
                        self.apgd.loss = 'dlr-targeted'
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y, attack_all_emps=attack_all_emps,
                                                              best_loss=best_loss)  # cheap=True
                    elif attack == 'apgd-adp-dlr-targeted':
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        assert target is not None
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss, target=y_target)  # cheap=True
                    elif attack == 'apgd-adp-dlr_in-targeted':
                        self.apgd.loss = 'adp-dlr_in'
                        self.apgd.seed = self.get_seed()
                        assert target is not None
                        adv_curr = self.apgd.perturb(x, y, attack_all_emps=attack_all_emps,
                                                     best_loss=best_loss, target=y_target)  # cheap=True
                    # elif attack == 'apgd-adp-dlr_in-targeted':
                    #     # targeted apgd
                    #     self.apgd.loss = 'adp-dlr_in-targeted'
                    #     self.apgd_targeted.seed = self.get_seed()
                    #     adv_curr = self.apgd_targeted.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss) #cheap=True
                    # elif attack == 'apgd-adp-dlr_in-out-targeted':
                    #     # targeted apgd
                    #     self.apgd.loss = 'adp-dlr_in-out-targeted'
                    #     self.apgd_targeted.seed = self.get_seed()
                    #     adv_curr = self.apgd_targeted.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    # elif attack == 'apgd-adp-dlr_in-v-targeted':
                    #     # targeted apgd
                    #     self.apgd.loss = 'adp-dlr_in-v-targeted'
                    #     self.apgd_targeted.seed = self.get_seed()
                    #     adv_curr = self.apgd_targeted.perturb(x, y, attack_all_emps=attack_all_emps, best_loss=best_loss)  # cheap=True
                    else:
                        raise ValueError('Attack not supported')

                    output = self.get_logits(adv_curr)
                    if self.data_type=='in':
                        false_batch = ~self.get_acc_flags(output, y)
                        non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                        robust_flags[non_robust_lin_idcs] = False
                        x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                        if self.verbose:
                            num_non_robust_batch = torch.sum(false_batch)
                            self.logger.log('{} - {}/{} - {} out of {} successfully perturbed'
                                            .format(attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))
                    else:
                        x_adv[batch_datapoint_idcs, :] = adv_curr
                        if self.verbose:
                            with torch.no_grad():
                                msp_nat = F.softmax(self.model(x), dim=1)[:, :self.num_in_classes].max(dim=1)[0]
                                msp_adv = F.softmax(self.model(x_adv[batch_datapoint_idcs, :]), dim=1)[:,
                                          :self.num_in_classes].max(dim=1)[0]
                                print('msp_nat:{}, msp_adv:{}'.format(msp_nat.mean().item(), msp_adv.mean().item()))

                if self.verbose and self.data_type == 'in':
                    robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                    self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s)'
                                    .format(attack.upper(), robust_accuracy, time.time() - startt))

            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().view(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).view(x_orig.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - x_orig).abs().view(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.log('robust accuracy: {:.2%}'.format(robust_accuracy))
        
        return x_adv
        
    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            if self.data_type == 'in':
                acc += (output[:, :self.num_in_classes].max(1)[1] == y).float().sum()
            else:
                if self.num_in_classes == output.size(1) or (self.num_out_classes == 0 and self.num_v_classes == 0):
                    acc += (output[:, :self.num_in_classes].max(1)[1] == y).float().sum()
                elif self.num_out_classes > 0:
                    confs, _ = torch.max(output[:, :self.num_in_classes], dim=1)
                    out_confs = torch.sum(output[:, self.num_in_classes:self.num_in_classes + self.num_out_classes],
                                          dim=1)
                    correct_batch_idcs = out_confs > confs
                    acc += correct_batch_idcs.float().sum()
                elif self.num_v_classes > 0:
                    acc += ((output[:, self.num_in_classes:self.num_in_classes + self.num_v_classes].max(1)[
                                 1] + self.num_in_classes) == y).float().sum()
                else:
                    raise ValueError('plz check self.num_in_classes:{}, self.num_out_classes:{}, '
                                     'self.num_v_classes:{}'.format(self.num_in_classes, self.num_out_classes,
                                                                    self.num_v_classes))
        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))
        
        return acc.item() / x_orig.shape[0]
        
    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, attack_all_emps=False, best_loss=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        
        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False
        
        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            adv[c] = self.run_standard_evaluation(x_orig, y_orig, bs=bs, attack_all_emps=attack_all_emps, best_loss=best_loss)
            if verbose_indiv:    
                acc_indiv  = self.clean_accuracy(adv[c], y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.log('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv,  time.time() - startt))
        
        return adv
        
    def set_version(self, version='standard'):
        if self.verbose:
            print('setting parameters for {} version'.format(version))
        
        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            #self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
        
        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, self.norm))
        
        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20

