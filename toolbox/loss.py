import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from toolbox.lavaszSoftmax import lovasz_softmax
# med_frq = [0.000000, 0.452448, 0.637584, 0.377464, 0.585595,
#            0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
#            2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
#            0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
#            1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
#            4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
#            3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
#            0.750738, 4.040773, 2.154782, 0.771605, 0.781544, 0.377464]

class lovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=None):
        super(lovaszSoftmax, self).__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.classes = classes
    def forward(self, output, target):
        if not isinstance(output, tuple):
            output = (output,)
        loss = 0
        for item in output:

            h, w = item.size(2), item.size(3)
            # 变换大小需要4维
            label = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            logits = F.softmax(item, dim=1)
            loss += lovasz_softmax(logits, label.squeeze(1), ignore=self.ignore_index,per_image=self.per_image,classes=self.classes)
        return loss/len(output)

class MscCrossEntropyLoss(nn.Module):
    #
    def __init__(self, weight=None, ignore_index=-100, reduction='mean',gate_gt=None):
        super(MscCrossEntropyLoss, self).__init__()

        self.weight = weight
        self.gate_gt = gate_gt
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        # 如果标签是RGB图像，我们只取其中一个通道

        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        # weight = [0.2,0.4,0.6,0.8]

        # h,w = target.size()[1:]

        for item in input:
            h, w = item.size(2), item.size(3)
            # h, w = item.size(1), item.size(2)

            # target = target.permute(0, 3, 1, 2)
            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            # print('ta0', target[0].shape)
            # target = target[0]
            # print('tar', target[0].shape)
            # print('item:', item.shape)
            # item_target = F.interpolate(target[0].unsqueeze(1).float(), size=(h, w))
            # print('item', item_target.shape)
            # item_target = F.interpolate(item, size=(h, w))



            # loss += F.cross_entropy(item, item_target.long(), weight=self.weight,
            #             ignore_index=self.ignore_index, reduction=self.reduction)
            loss += F.cross_entropy(item, item_target.squeeze(1).long(), weight=self.weight,
                                    ignore_index=self.ignore_index, reduction=self.reduction)
            # loss += F.cross_entropy(item_target, target.long(), weight=self.weight,
            #                         ignore_index=self.ignore_index, reduction=self.reduction)


            #对输入的一个batch求loss的平均
        return loss / len(input)


if __name__ == '__main__':
    x = torch.randn(2,2)
    print(x)
    out = x.mean(1)
    # import torch
    # ll = 'layer3_1 '
    # out = ll.split('_1')[0]+ll.split('_1')[1]
    print(out)
    # depth = torch.randn(6,3,480,640)
    # score = torch.Tensor(6,1)
    # print(score.shape)
    # print(score)
    # score = score[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,3,480,640)
    # # out = torch.mul(depth,score[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,3,480,640))
    # print(score.shape)
    # print(score)
    # torch.randn(6,3,480,640)
    # print(out)
    # out = out.view(3,480,640)
    # print(out)

    # predict = torch.randn((2, 21, 512, 512))
    # gt = torch.randint(0, 255, (2, 512, 512))

    # loss_function = MscCrossEntropyLoss()
    # result = loss_function(predict, gt)
    # print(result)
