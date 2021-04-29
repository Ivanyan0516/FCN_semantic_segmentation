import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import argparse
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR
from utils import FCN8s, FCN16s, FCN32s, FCN8x, train_transform, voc_dataset, label_accuracy_score, label2image
import matplotlib.pyplot as plt
import random
# 使用tensorboard
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description = 'FCN')
parser.add_argument('--bs', type = int, default = 5)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--epochs',type = int, default = 32)
parser.add_argument('--result_path', type = str, default = './result.txt', help = 'path to print result')
parser.add_argument('--model_path', type = str, default = './models/best_model.pth', help = 'path to save model weights')
parser.add_argument('--summarywriter_dir', type = str, default = './runs', help = 'path to save model weights')
parser.add_argument('--lr_decay_choice', default = 'Cosine', type = str, help = 'lr scheduler, eg. Exponential, multistep')
parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='decay rate')
parser.add_argument('--model', required = True)
parser.add_argument('--optim', required = True)
args = parser.parse_args()

torch.manual_seed(78)

SumWriter = SummaryWriter(log_dir = args.summarywriter_dir)
data_path = './data' #数据集路径
result_path = args.result_path #结果打印路径
model_path = args.model_path #最优模型保存路径
bs = args.bs
lr = args.lr
epoch = args.epochs
numclasses = 21
crop = (256,256)
ratio = 0.9 #训练集比例
use_gpu = torch.cuda.is_available()

if os.path.exists(result_path):
    os.remove(result_path)

#构建数据集
dataset = voc_dataset(root=data_path,transfrom=train_transform,crop_size=crop)
train_size = int(len(dataset)*ratio)
train_data, val_data = random_split(dataset, [train_size, len(dataset) - train_size])
train_dataloader = DataLoader(train_data, batch_size = args.bs, shuffle = True)
val_dataloader = DataLoader(val_data, batch_size=args.bs, shuffle = False)
 
#构建网络
Net = None
if args.model == 'fcn8s':
    Net = FCN8s
elif args.model == 'fcn8x':
    Net = FCN8x
elif args.model == 'fcn16':
    Net = FCN16s
elif args.model == 'fcn32':
    Net = FCN32s
else:
    assert Net is not None, f'model {args.model} not available'

net = Net(numclasses)

# 选择优化器
if args.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr = args.lr)
elif args.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr = args.lr)
criterion = nn.NLLLoss()

if use_gpu:
    net.cuda()
    criterion=criterion.cuda()

# 选择学习率下降方法
if args.lr_decay_choice == 'Exponential':
    scheduler = ExponentialLR(optimizer, gamma = args.lr_decay_rate, last_epoch = -1)
elif args.lr_decay_choice == 'Multistep':
    scheduler = MultiStepLR(optimizer, milestones = [10, 20, 30], gamma = 0.2)  # learning rates milestones=[30, 60, 90]
elif args.lr_decay_choice == 'Cosine':
    scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 0, last_epoch = -1)
else:
    raise Exception('ERROR: no specific criterion ...')
    
#训练验证
def train():
    best_score = 0.0
    for e in range(args.epochs):
        scheduler.step(e)
        net.train()
        train_loss = 0.0
        label_true = torch.LongTensor()
        label_pred = torch.LongTensor()
        for i,(batchdata, batchlabel) in enumerate(train_dataloader):
            '''
            batchdata:[b,3,h,w] c=3
            batchlabel:[b,h,w] c=1 直接去掉了
            '''
            if use_gpu:
                batchdata,batchlabel = batchdata.cuda(),batchlabel.cuda()

            output = net(batchdata)
            output = F.log_softmax(output, dim = 1)
            loss = criterion(output, batchlabel)

            pred = output.argmax(dim = 1).squeeze().data.cpu()
            real = batchlabel.data.cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()*batchlabel.size(0)
            label_true = torch.cat((label_true,real), dim = 0)
            label_pred = torch.cat((label_pred,pred),dim = 0)
#             print('iteration {} finished'.format(i))
        train_loss /= len(train_data)
        acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(label_true.numpy(), label_pred.numpy(), numclasses)

        print('\n epoch:{}, train_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}, LR:{:.6f}'.format(
            e + 1, train_loss, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[0]['lr']))
        
        SumWriter.add_scalar("train loss", train_loss, global_step = e + 1)
        SumWriter.add_scalar("train acc", acc, global_step = e + 1)
        SumWriter.add_scalar("train acc_cls", acc_cls, global_step = e + 1)
        SumWriter.add_scalar("train mean_iu", mean_iu, global_step = e + 1)
        SumWriter.add_scalar("train fwavacc", fwavacc, global_step = e + 1)
        
        with open(result_path, 'a') as f:
            f.write('\n epoch:{}, train_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}'.format(
                e + 1, train_loss,acc, acc_cls, mean_iu, fwavacc))
        # validation
        net.eval()
        val_loss = 0.0
        val_label_true = torch.LongTensor()
        val_label_pred = torch.LongTensor()
        with torch.no_grad():
            for i, (batchdata,batchlabel) in enumerate(val_dataloader):
                if use_gpu:
                    batchdata,batchlabel = batchdata.cuda(),batchlabel.cuda()

                output = net(batchdata)
                output = F.log_softmax(output, dim = 1)
                loss = criterion(output,batchlabel)

                pred = output.argmax(dim = 1).squeeze().data.cpu()
                real = batchlabel.data.cpu()

                val_loss += loss.cpu().item() * batchlabel.size(0)
                val_label_true = torch.cat((val_label_true, real), dim = 0)
                val_label_pred = torch.cat((val_label_pred, pred), dim = 0)

            val_loss /= len(val_data)
            val_acc, val_acc_cls, val_mean_iu, val_fwavacc = label_accuracy_score(val_label_true.numpy(),
                                                                                val_label_pred.numpy(),numclasses)
        print('\n epoch:{}, val_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}'.format(
            e+1,val_loss,val_acc, val_acc_cls, val_mean_iu, val_fwavacc))
        SumWriter.add_scalar("val loss", val_loss, global_step = e + 1)
        SumWriter.add_scalar("val acc", val_acc, global_step = e + 1)
        SumWriter.add_scalar("val acc_cls", val_acc_cls, global_step = e + 1)
        SumWriter.add_scalar("val mean_iu", val_mean_iu, global_step = e + 1)
        SumWriter.add_scalar("val fwavacc", val_fwavacc, global_step = e + 1)
        
        with open(result_path, 'a') as f:
            f.write('\n epoch:{}, val_loss:{:.4f}, acc:{:.4f}, acc_cls:{:.4f}, mean_iu:{:.4f}, fwavacc:{:.4f}'.format(
            e + 1,val_loss,val_acc, val_acc_cls, val_mean_iu, val_fwavacc))

        score = (val_acc_cls+val_mean_iu)/2
        if score > best_score:
            best_score = score
            torch.save(net.state_dict(), model_path)

#加载网络进行测试
def evaluate():
    net.load_state_dict(torch.load(model_path))
    index = random.randint(0, len(dataset) - 1)
    val_image, val_label = dataset[index]

    out = net(val_image.unsqueeze(0).cuda())
    pred = out.argmax(dim = 1).squeeze().data.cpu().numpy()
    label = val_label.data.numpy()
    val_pred, val_label = label2image(numclasses)(pred, label)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(val_image.numpy().transpose(1, 2, 0).astype('uint8'))
    ax[1].imshow(val_label)
    ax[2].imshow(val_pred)
    plt.show()
  

if __name__ == '__main__':
    train()
