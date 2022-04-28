import torch
import torch.utils.data as data
import cv2
import os
import os.path
import random
import numpy as np


class yoloDataset(data.Dataset):
    image_size = 448

    def __init__(self, root, list_file, train, transform):
        print('loading annotations')

        self.root = root
        self.train = train
        self.transform = transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.S = 7  # grid number 7*7 normally
        self.B = 2  # bounding box number in each grid
        self.C = 1 # how many classes 改了
        self.mean = (123, 117, 104)  # RGB

        # if isinstance(list_file, list):
        #     # Cat multiple list files together.
        #     # This is especially useful for voc07/voc12 combination.
        #     # 将voc2007和voc2012两个数据集的标签整合为一
        #     tmp_file = '/tmp/listfile.txt'
        #     os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
        #     list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])         # 存储图片的名字
            num_boxes = (len(splited) - 1) // 5    # 每一幅图片里面有多少个bbox
            box = []
            label = []
            for i in range(num_boxes):
                x = float(splited[1 + 5 * i])
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                c = splited[5 + 5 * i]            # 代表物体的类别，即是20种物体里面的哪一种
                box.append([x, y, x2, y2])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root + fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        if self.train:  # 数据增强里面的各种变换用torch自带的transform是做不到的，因为对图片进行旋转、随即裁剪等会造成bbox的坐标也会发生变化，所以需要自己来定义数据增强
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img, boxes, labels = self.randomShift(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)         # 坐标归一化处理，为了方便训练
        img = self.BGR2RGB(img)                                                                           # because pytorch pretrained model use RGB
        img = self.subMean(img, self.mean)                          # 减去均值
        img = cv2.resize(img, (self.image_size, self.image_size))   # 将所有图片都resize到指定大小
        target = self.encoder(boxes, labels)                        # 将图片标签编码到7x7*30的向量

        for t in self.transform:
            img = t(img)

        return img, target

    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        grid_num = 7
        target = torch.zeros((grid_num, grid_num, 11))     # 这里改了 原来是30
        cell_size = 1. / grid_num                         # 每个格子的大小

        # 右下坐标        左上坐标
        # x2,y2           x1,y1
        wh = boxes[:, 2:] - boxes[:, :2]

        # 物体中心坐标集合
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(cxcy.size()[0]):
            # 物体中心坐标
            cxcy_sample = cxcy[i]

            # 指示落在那网格，如[0,0]
            ij = (cxcy_sample / cell_size).ceil() - 1  # 中心点对应格子的坐标
            #    0，1  2  3   4    5，6   7  8   9，     10+
            # [中心坐标,长,宽,置信度,中心坐标,长,宽,置信度, 20个类别] x 7x7   因为一个框预测两个物体

            # 第一个框的置信度
            target[int(ij[1]), int(ij[0]), 4] = 1

            # 第二个框的置信度
            target[int(ij[1]), int(ij[0]), 9] = 1

            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1      # 类别

            # xy为归一化后网格的左上坐标---->相对整张图
            xy = ij * cell_size

            # 物体中心相对左上的坐标 ---> 坐标x,y代表了预测的bounding
            # box的中心与栅格边界的相对值
            delta_xy = (cxcy_sample - xy) / cell_size  # 其实就是offset

            # (1) 每个小格会对应B(2)个边界框，边界框的宽高范围为全图，表示以该小格为中心寻找物体的边界框位置。
            # (2) 每个边界框对应一个分值，代表该处是否有物体及定位准确度
            # (3) 每个小格会对应C个概率值，找出最大概率对应的类别P(Class|object)，并认为小格中包含该物体或者该物体的一部分。

            # 坐标w,h代表了预测的bounding box的width、height相对于整幅图像width,height的比例
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy

            # 每一个网格有两个边框
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]           # 长宽
            # 中心坐标偏移
            # 由此可得其实返回的中心坐标其实是相对左上角顶点的偏移，因此在进行预测的时候还需要进行解码
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        # 平移变换
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):,
                                  int(shift_x):,
                                  :] = bgr[:height - int(shift_y),
                                           :width - int(shift_x),
                                           :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y),
                                  int(shift_x):,
                                  :] = bgr[-int(shift_y):,
                                           :width - int(shift_x),
                                           :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width +
                                  int(shift_x), :] = bgr[:height -
                                                         int(shift_y), -
                                                         int(shift_x):, :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(
                    shift_x), :] = bgr[-int(shift_y):, -int(shift_x):, :]

            shift_xy = torch.FloatTensor(
                [[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor(
                [[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor(
                [[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if(len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im


def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    file_root = 'F:/face_datas/VOC/VOCdevkit/VOC2012/JPEGImages/'
    train_dataset = yoloDataset(
        root=file_root,
        list_file='./voc2012.txt',
        train=True,
        transform=[
            transforms.ToTensor()])
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0)
    train_iter = iter(train_loader)
    for i in range(100):
        img, target = next(train_iter)
        print(img, target)


if __name__ == '__main__':
    main()
