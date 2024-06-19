import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable




class WeightedSoftDiceLoss(torch.nn.Module):
    def __init__(self, **_):
        super(WeightedSoftDiceLoss, self).__init__()

    @staticmethod
    def forward(logits, labels, weights, **_):
        probs = torch.sigmoid(logits)
        num   = labels.size(0)
        w     = weights.view(num,-1).to(logits.device) 
        w2    = w*w
        m1    = probs.view(num,-1)
        m2    = labels.view(num,-1).to(logits.device) 
        intersection = (m1 * m2)
        score = 2. * ((w2*intersection).sum(1)+1) / ((w2*m1).sum(1) + (w2*m2).sum(1)+1)
        score = 1 - score.sum()/num
        return score

class WeightedBCELoss2d(nn.Module):
    def __init__(self, **_):
        super(WeightedBCELoss2d, self).__init__()

    @staticmethod
    def forward(logits, labels, weights, **_):
        w = weights.view(-1).to(logits.device)      # (-1 operation flattens all the dimensions)
        z = logits.view(-1)                         # (-1 operation flattens all the dimensions)
        t = labels.view(-1).to(logits.device)       # (-1 operation flattens all the dimensions)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/w.sum()
        return loss


class  BCEDicePenalizeBorderLoss(nn.Module):
    def __init__(self, kernel_size=55, dice_ratio = 0.5, **_):
        super(BCEDicePenalizeBorderLoss, self).__init__()
        self.bce = WeightedBCELoss2d()
        self.dice = WeightedSoftDiceLoss()
        self.kernel_size = kernel_size
        self.dice_ratio = dice_ratio

    def to(self, device):
        super().to(device=device)
        self.bce.to(device=device)
        self.dice.to(device=device)

    def forward(self, logits, labels, **_):
        a = F.avg_pool2d(labels, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        ind = a.ge(0.01) * a.le(0.99)
        ind = ind.float()
        ind = ind.to(device=logits.device)
        weights = torch.ones(a.size()).to(device=logits.device)

        w0 = weights.sum()
        weights = weights + ind * 2
        w1 = weights.sum()
        weights = weights / w1 * w0

        bce_weight = (1-self.dice_ratio) / 0.5
        dice_weight = (self.dice_ratio) / 0.5

        loss = (bce_weight * self.bce(logits, labels, weights)) + (dice_weight * self.dice(logits, labels, weights))
        return loss


def norm255(data):
    maxval = np.max(data)
    minval = np.min(data)
    
    return (((data - minval) / (maxval - minval))*255).astype(np.uint8)





def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss
    
def dice_coeff_np(score, target):
    score = score.astype(np.float32)
    target = target.astype(np.float32)
    smooth = 1e-5
    intersect = np.sum(score * target)
    y_sum = np.sum(target * target)
    z_sum = np.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return loss


def dice_score(score,target):
    score = score.astype(np.float32)
    target = target.astype(np.float32)

    intersect = np.sum(score * target)
    y_sum = np.sum(target * target)
    z_sum = np.sum(score * score)
    try:
        dc = (2 * intersect) / (z_sum + y_sum)
    except ZeroDivisionError:
        dc = 1.0
    return dc



def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        inputs = torch.sigmoid(inputs)       

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    

def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


