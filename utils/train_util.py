import matplotlib; matplotlib.use('Agg')
import torch
from torch.optim.lr_scheduler import _LRScheduler

def get_optimizer(args, feature_extractor, classifier):
    optimizer_params = args.optimizer_params

    optimizer_params = {
        k: float(v) if isinstance(v, (int, float, str)) else v
        for k, v in optimizer_params.items()
    }
    initial_lr = optimizer_params["initial_lr"]
    weight_decay = optimizer_params["weight_decay_values"]
    optimizer = torch.optim.Adam(
        [
            {'params': classifier.parameters(), 'lr': initial_lr, 'weight_decay': weight_decay}
        ]
    )
    return optimizer

def warmup_scheduler(optimizer, current_epoch, warmup_epochs, base_lr):
    if current_epoch < warmup_epochs:
        warmup_factor = current_epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * warmup_factor

def get_lr_scheduler(args, optimizer, iters_per_epoch=-1):
    scheduler_params = args.scheduler_params
    T_max = int(scheduler_params["T_max"])
    eta_min = float(scheduler_params["eta_min"])
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor = pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        assert factor >= 0, 'error in lr_scheduler'
        return [base_lr * factor for base_lr in self.base_lrs]
