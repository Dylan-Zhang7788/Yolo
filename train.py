
from itertools import count
from tracemalloc import stop
from dataset import yoloDataset
from yoloLoss import yoloLoss
from resnet import resnet50
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import hiddenlayer as hl
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

history1 = hl.History()
history2=hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
canvas2 = hl.Canvas()
stop_count=0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
file_root_train = '/home/xcy/zhangdi_ws/seg/train_dataset/'
file_root_val='/home/xcy/zhangdi_ws/seg/train_dataset/'
batch_size = 24
learning_rate = 0.001
num_epochs = 100

train_dataset = yoloDataset(
    root=file_root_train, list_file=
        '/home/xcy/zhangdi_ws/seg/train_dataset.txt', train=True, transform=[
            transforms.ToTensor()])
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
val_dataset = yoloDataset(
    root=file_root_val,
    list_file='/home/xcy/zhangdi_ws/seg/train_dataset.txt',
    train=False,
    transform=[
        transforms.ToTensor()])
val_loader = DataLoader(
    val_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=0)

print('the train_dataset has %d images' % (len(train_dataset)))
print('the val_dataset has %d images' % (len(val_dataset)))
print('the batch_size is %d' % (batch_size))
print('loading network structure...')

net = resnet50()
net = net.to(device)
print('load resnet structure...')
# print(net)

print('load pre_trained model...')
resnet = models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()

op = net.state_dict()
for k in new_state_dict.keys():
    # print(k)
    if k in op.keys() and not k.startswith('fc'):  # startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
        # print('yes')
        op[k] = new_state_dict[k]
net.load_state_dict(op)

if False:
    net.load_state_dict(torch.load('best.pth'))
print('testing the cuda device here')
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

Yolo = yoloLoss(7, 2, 5, 0.5)

net.train()
# different learning rate
params = []
params_dict = dict(net.named_parameters())

for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr':learning_rate * 1}] 
    else:
        params += [{'params': [value], 'lr':learning_rate}]
optimizer = torch.optim.SGD(
    params,
    lr=learning_rate,
    momentum=0.9,
    weight_decay=5e-4)

# torch.multiprocessing.freeze_support()
best_test_loss =1000000
time=0

for epoch in range(num_epochs):
    net.train()
    if epoch == 30:
        learning_rate = 0.0005
    if epoch == 40:
        learning_rate = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0.
    train_loss_epoch=0.
    for i, (images, target) in enumerate(train_loader):
        images, target = images.cuda(), target.cuda()
        pred = net(images)
        loss = Yolo(pred, target)
        total_loss += loss.item()
        train_loss_epoch=total_loss*images.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % 
            (epoch +1, num_epochs,i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))
            time+=1
            history1.log(time, train_loss=total_loss / (i + 1))
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
    train_loss = train_loss_epoch / 17117
    if best_test_loss > train_loss:
        best_test_loss = train_loss
        stop_count=0
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(), 'best.pth')
    else:
        stop_count+=1
        print('EarlyStopping counter:',stop_count, 'out of 8')
        if stop_count>8: break

torch.save(net.state_dict(), 'yolo.pth')

        

    # validation_loss = 0.0
    # net.eval()
    # for i, (images, target) in enumerate(val_loader):
    #     images, target = images.cuda(), target.cuda()
    #     pred = net(images)
    #     loss = Yolo(pred, target)
    #     validation_loss += loss.item()
    # validation_loss /= len(val_loader)
    # history2.log(epoch +1,val_loss=validation_loss)
    # canvas2.draw_plot(history1["val_loss"])


torch.save(net.state_dict(), 'yolo.pth')


