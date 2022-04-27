import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import ByteTensor

class yoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):  # 为了更重视8维的坐标预测，给这些算是前面赋予更大的loss weight
        # 对于有物体的记为λcoord，在pascal VOC训练中取5，对于没有object的bbox的confidence loss，前面赋予更小的loss weight 记为 λnoobj
        # 在pascal VOC训练中取0.5
        # 有object的bbox的confidence loss (上图红色框) 和类别的loss （上图紫色框）的loss weight正常取1

        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = bool(l_coord)
        self.l_noobj = bool(l_noobj)

    def compute_iou(self, box1, box2):
        # iou的作用是，当一个物体有多个框时，选一个相比ground truth最大的执行度的为物体的预测，然后将剩下的框降序排列，如果后面的框中有与这个框的iou大于一定的阈值时则将这个框舍去（这样就可以抑制一个物体有多个框的出现了），目标检测算法中都会用到这种思想。
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        N = box1.size(0)
        M = box2.size(0)

        # torch.max返回每一个位置上最大的数字
        lt = torch.max(
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, :2].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),
        )

        rb = torch.min(
            # [N,2] -> [N,1,2] -> [N,M,2]
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),
            # [M,2] -> [1,M,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]---预测对应的格式
        target_tensor: (tensor) size(batchsize,S,S,30) --- 标签的准确格式
        '''
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:, :, :, 4] > 0   # 具有目标标签的索引

        noo_mask = target_tensor[:, :, :, 4] == 0  # 不具有目标的标签索引
        # 得到含物体的坐标等信息
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        # 得到不含物体的坐标等信息
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1, 30)
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:, 10:]  # [x2,y2,w2,h2,c2]

        coo_target = target_tensor[coo_mask].view(-1, 30)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]
        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1, 30)
        noo_target = target_tensor[noo_mask].view(-1, 30)
        noo_pred_mask = ByteTensor(noo_pred.size()).cuda()
        noo_pred_mask.zero_()
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        noo_pred_mask = noo_pred_mask.bool()
        noo_pred_c = noo_pred[noo_pred_mask]  # noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(
            noo_pred_c,
            noo_target_c,
            size_average=False)  # 对应的位置做均方误差

        # compute contain obj loss
        coo_response_mask = ByteTensor(box_target.size()).cuda()
        coo_response_mask.zero_()
        coo_not_response_mask = ByteTensor(box_target.size()).cuda()
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()

        # 预测值，有多个box的话那么就取一个最大的box，出来就可以了其他的不要啦
        for i in range(0, box_target.size()[0], 2):  # choose the best iou box ， box1 是预测的 box2 是我们提供的
            box1 = box_pred[i:i + 2]
            box1_xyxy = torch.autograd.Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:, :2] = box1[:, :2] / 14. - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, :2] / 14. + 0.5 * box1[:, 2:4]
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = torch.autograd.Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] / 14. - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / 14. + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - max_index] = 1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = max_iou.data.cuda()

        coo_response_mask = coo_response_mask.bool()
        coo_not_response_mask = coo_not_response_mask.bool()

        box_target_iou = torch.autograd.Variable(box_target_iou).cuda()
        # 1.response loss，iou符合的
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)

        contain_loss = F.mse_loss(
            box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), size_average=False)
        # 2.not response loss iou不符合的
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        # not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)

        # I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(
            box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=False)

        # 3.class loss
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)

        return (self.l_coord * loc_loss + 2 * contain_loss +
                not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N

