import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import models
from losses import LPSL
import resnet
from torch.autograd import Variable
from PIL import ImageFile
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import argparse

from dataset import CIFAR100Dataset,CIFAR100DatasetAL
from acts import Acts

parser = argparse.ArgumentParser(description="CIFAR100 LPSL ACTS Running")
parser.add_argument('--gpu', default=0, type=int, help="gpu id")

# LPSL parameters
parser.add_argument('--_lambda', default=1, type=float)
parser.add_argument('--eta', default=0.1, type=float)

# Acts parameters
parser.add_argument('--q_budget', default=2000, type=int)
parser.add_argument('--delta', default="F", type=str)
parser.add_argument('--T1', default=0.1, type=float)

# Training parameters
parser.add_argument('--bs', default=64, type=int, help="batch size")
parser.add_argument('--E1', default=21, type=int)
parser.add_argument('--E2', default=30, type=int)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

# 设置全局参数
BATCH_SIZE = args.bs
E1 = args.E1
E2 = args.E2 # 训练EPOCH数量 = E1xE2
n_classes = 128 # total class number
HLevel = [1,2,3] # Level 3
lambda1=args._lambda 
lambda2=args.eta 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loss_file = "/home/wyx/vscode_projects/hier/models/CIFAR100/acts_train_loss_lpsl_acts_delta{}T{}.csv".format(args.delta,args.T1)
val_loss_file = "/home/wyx/vscode_projects/hier/models/CIFAR100/acts_val_loss_lpsl_acts_delta{}T{}.csv".format(args.delta,args.T1)
test_acc_file = "/home/wyx/vscode_projects/hier/models/CIFAR100/acts_test_acc_lpsl_acts_delta{}T{}.csv".format(args.delta,args.T1)
best_model_path = "/home/wyx/vscode_projects/hier/models/CIFAR100/acts_best_model_lpsl_delta{}T{}.pth".format(args.delta,args.T1)
acts_out_path = "/home/wyx/vscode_projects/hier/models/CIFAR100/acts_Q_delta{}T{}.csv".format(args.delta,args.T1)
with open(train_loss_file, "w") as f:
    f.write("Epoch,Loss,Level1CE,Level2CE,Level3CE,LPS,HVP\n")
with open(val_loss_file, "w") as f:
    f.write("Epoch,Loss,Level1CE,Level2CE,Level3CE,LPS,HVP\n")
with open(test_acc_file, "w") as f:
    f.write("Epoch,Level1-ACC,Level2-ACC,Level3-ACC\n")

# 读取数据
dataset_L,dataset_U = CIFAR100DatasetAL.createLURandomly(HLevel=HLevel,image_size=32)
dataset_test = CIFAR100Dataset(train=False, HLevel=HLevel, image_size=32)
train_loader = torch.utils.data.DataLoader(dataset_L, batch_size=BATCH_SIZE, shuffle=True,num_workers=16)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False,num_workers=16)

# loss 类型
criterion = LPSL(CIFAR100Dataset.CLASS_HIERARCHY,lambda1,lambda2)

#model = models.resnet18(pretrained=False)
#model.conv1 = nn.Conv2d(3, model.inplanes, kernel_size=3, stride=2, padding=3, bias=False)
#model.fc = nn.Linear(512, n_classes)
# 初始化 https://androidkt.com/initialize-weight-bias-pytorch/
#model.fc.bias.data.fill_(-np.log(n_classes-1))

# Using the pytorch-cifar10 implementation
# The 1st convolution layer for CIFAR-10 is 3x3
model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=n_classes)
model.to(DEVICE)

_acts_ = Acts(dataset_L, dataset_U, 
              model, 
              args.q_budget, 
              DEVICE, 
              CIFAR100Dataset.CLASS_HIERARCHY, 
              delta=args.delta, 
              m=E1-1, 
              T1=args.T1, 
              Tm=0.001,
              DEBUG=True,
              exp_out_file=acts_out_path)

#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# 定义训练过程
def train(epoch):
    model.train()
    # Reset the loss value
    sum_loss = 0
    criterion.ce = [0]*3
    criterion.lps = 0
    criterion.hvp = 0
    total_num = len(train_loader.dataset)
    print("NumSamples:", total_num, "NumBatch: ", len(train_loader))
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(DEVICE), Variable(target).to(DEVICE)
        output = model(data)
        loss = criterion.loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.data.item()
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item())) 
            print("Time: {:.2f} s".format(time.time() - start))
            start = time.time()
            
    ave_loss = sum_loss / len(train_loader)
    l1_ce = criterion.ce[0] / len(train_loader)
    l2_ce = criterion.ce[1] / len(train_loader)
    l3_ce = criterion.ce[2] / len(train_loader)
    lps = criterion.lps / len(train_loader)
    hvp = criterion.hvp / len(train_loader)
    print('Epoch:{}, Loss:{:.4f},Level1CE:{:.4f},Level2CE:{:.4f},Level3CE:{:.4f},LPS:{:.4f},HVP:{:.4f}'.format(epoch, ave_loss,l1_ce,l2_ce,l3_ce,lps,hvp))
    with open(train_loss_file, "a") as f:
        f.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(epoch,ave_loss,l1_ce,l2_ce,l3_ce,lps,hvp))
    
#验证过程
def val(epoch):
    model.eval()
    real1_labels = []
    real2_labels = []
    real3_labels = []
    predict1_labels = []
    predict2_labels = []
    predict3_labels = []
    sum_loss = 0
    criterion.ce = [0]*3
    criterion.lps = 0
    criterion.hvp = 0
    total_num = len(test_loader.dataset)
    print("Test samples Num: {}, Batch Num: {}".format(total_num, len(test_loader)))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(DEVICE), Variable(target).to(DEVICE)
            output = model(data)
            loss = criterion.loss(output, target)
            sum_loss += loss.data.item()
            _, pred1 = torch.max(output.data[..., :8], dim=1)
            _, real1 = torch.max(target.data[..., :8], dim=1)
            _, pred2 = torch.max(output.data[..., 8:28], dim=1)
            _, real2 = torch.max(target.data[..., 8:28], dim=1)
            _, pred3 = torch.max(output.data[..., 28:], dim=1)
            _, real3 = torch.max(target.data[..., 28:], dim=1)
            
            real1_labels.extend(list(real1.cpu().numpy()))
            predict1_labels.extend(list(pred1.cpu().numpy()))
            real2_labels.extend(list(real2.cpu().numpy()))
            predict2_labels.extend(list(pred2.cpu().numpy()))
            real3_labels.extend(list(real3.cpu().numpy()))
            predict3_labels.extend(list(pred3.cpu().numpy()))
            
        acc1,acc2,acc3 = 0,0,0
        assert len(real1_labels) == len(real2_labels) == len(real3_labels) ==  \
            len(predict1_labels) == len(predict2_labels) == len(predict3_labels)
        for i in range(len(real1_labels)):
            # Level-3
            if int(real1_labels[i]) == int(predict1_labels[i]):
                acc1 += 1
            if int(real2_labels[i]) == int(predict2_labels[i]):
                acc2 += 1
            if int(real3_labels[i]) == int(predict3_labels[i]):
                acc3 += 1
                
        assert len(real1_labels) == total_num
        acc1 /= total_num
        acc2 /= total_num   
        acc3 /= total_num
        
        print('Epoch: {}, Level-1 CE ACC: {:.2f}%'.format(epoch, 100 * acc1))
        print('Epoch: {}, Level-2 CE ACC: {:.2f}%'.format(epoch, 100 * acc2))
        print('Epoch: {}, Level-3 CE ACC: {:.2f}%'.format(epoch, 100 * acc3))
        with open(test_acc_file, "a") as f:
            f.write("{},{:.4f},{:.4f},{:.4f}\n".format(epoch,acc1,acc2,acc3))
            
        ave_loss = sum_loss / len(test_loader)
        l1_ce = criterion.ce[0] / len(test_loader)
        l2_ce = criterion.ce[1] / len(test_loader)
        l3_ce = criterion.ce[2] / len(test_loader)
        lps = criterion.lps / len(test_loader)
        hvp = criterion.hvp / len(test_loader)
        print('Epoch:{}, Loss:{:.4f},Level1CE:{:.4f},Level2CE:{:.4f},Level3CE:{:.4f},LPS:{:.4f},HVP:{:.4f}'.format(epoch, ave_loss,l1_ce,l2_ce,l3_ce,lps,hvp))
        with open(val_loss_file, "a") as f:
            f.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(epoch,ave_loss,l1_ce,l2_ce,l3_ce,lps,hvp))
            
        return acc1,acc2,acc3

# 训练
eval_losses = []
best_acc = 0
for epoch in range(1, E1*E2 + 1):
    print("Training....")
    train(epoch)
    print("Testing....")
    acc = val(epoch)
    if np.mean(acc) > best_acc:
        best_acc = np.mean(acc)
        torch.save(model.state_dict(), best_model_path)
    
    if epoch % E2 == 0:
        # Load the best model
        model.load_state_dict(torch.load(best_model_path))
        # Select samples and updating
        _acts_.select()
        _acts_.update_data()
        train_loader = torch.utils.data.DataLoader(_acts_.L, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    
    
    