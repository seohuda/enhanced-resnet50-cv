import torch
import numpy as np
import torch.nn as nn


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W - 1)
    bby1 = np.clip(cy - cut_h // 2, 0, H - 1)
    bbx2 = np.clip(cx + cut_w // 2, 1, W)
    bby2 = np.clip(cy + cut_h // 2, 1, H)

    return int(bbx1), int(bby1), int(bbx2), int(bby2)


class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets_a, targets_b, lam):
        log_probs = nn.functional.log_softmax(outputs, dim=1)
        loss_a = nn.functional.nll_loss(log_probs, targets_a)
        loss_b = nn.functional.nll_loss(log_probs, targets_b)
        return lam * loss_a + (1 - lam) * loss_b


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
