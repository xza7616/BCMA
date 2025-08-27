import torch.nn as nn
import torch.nn.functional as F
# from toolbox.loss.lovasz_loss import LovaszSoftmax
import torch


class MscCrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(MscCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        for item in input:
            h, w = item.size(2), item.size(3)
            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            loss += F.cross_entropy(item, item_target.squeeze(1).long(), weight=self.weight,
                        ignore_index=self.ignore_index, reduction=self.reduction)
        return loss / len(input)


# class MscLovaszSoftmaxLoss(nn.Module):
#     def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
#         super(MscLovaszSoftmaxLoss, self).__init__()
#         self.weight = weight
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#
#     def forward(self, input, target):
#         if not isinstance(input, tuple):
#             input = (input,)
#
#         loss = 0
#         for item in input:
#             h, w = item.size(2), item.size(3)
#             item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
#             loss += LovaszSoftmax(item, item_target.squeeze(1).long())
#         return loss / len(input)


class FocalLossbyothers(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=-100):
        super(FocalLossbyothers, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1-pt) ** self.gamma) * self.alpha * logpt
        return loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, preds, labels):
        labels = labels
        preds = preds
        intersection = (preds * labels).sum(axis=(2, 3))
        unior = (preds + labels).sum(axis=(2, 3))
        dice = (2 * intersection + 1) / (unior + 1)
        dice = torch.mean(1 - dice)
        return dice


class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1, resize_config=None, shuffle_config=None, transform_config=None,
                 warmup_config=None, earlydecay_config=None):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = resize_config
        self.shuffle_config = shuffle_config
        # print("self.shuffle", self.shuffle_config)
        self.transform_config = transform_config
        self.warmup_config = warmup_config
        self.earlydecay_config = earlydecay_config

        self.KLD = torch.nn.KLDivLoss(reduction='sum')


    def forward(self, x_student, x_teacher):
        # print("start kld")
        # if self.warmup_config:
        #     print("warm")
        #     self.warmup(n_iter)
        # if self.earlydecay_config:
        #     print("decay")
        #     self.earlydecay(n_iter)
        #
        # if self.resize_config:
        #     print("resize(")
        #     x_student, x_teacher = self.resize(x_student, gt), self.resize(x_teacher, gt)
        # if self.shuffle_config:
        #     print("shuffle")
        #     x_student, x_teacher = self.shuffle(x_student, x_teacher, n_iter)
        # if self.transform_config:
        #     print("transform")
        #     x_student, x_teacher = self.transform(x_student), self.transform(x_teacher)
        # print("hhh")

        x_student = F.log_softmax(x_student / self.tau, dim=-1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=-1)
        loss = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[-1])
        # print("self.alpha", self.alpha)
        loss = self.alpha * loss
        return loss


class CosLoss(nn.Module):
    def __init__(self):
        super(CosLoss, self).__init__()

    def forward(self, stu_map, tea_map):
        similiar = 1 - F.cosine_similarity(stu_map, tea_map, dim=1)
        loss = similiar.sum()
        return loss