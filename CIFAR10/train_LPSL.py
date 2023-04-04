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
from torch.autograd import Variable
from PIL import ImageFile
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import argparse

from dataset import CIFAR10Dataset

parser = argparse.ArgumentParser(description="CIFAR-10 LPSL Running")
parser.add_argument('--gpu', default=0, type=int, help="gpu id")
parser.add_argument('--_lambda', default=0, type=float)
parser.add_argument('--eta', default=0, type=float)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

# 设置全局参数
BATCH_SIZE = 64
c_epoch = 0
EPOCHS = 30 # 训练EPOCH数量
n_classes = 19 # total class number
HLevel = [1,2,3] # Level 3
lambda1=args._lambda 
lambda2=args.eta 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loss_file = "/home/wyx/vscode_projects/hier/models/CIFAR10/train_loss_lpsl_lambda{}eta{}.csv".format(lambda1,lambda2)
val_loss_file = "/home/wyx/vscode_projects/hier/models/CIFAR10/val_loss_lpsl_lambda{}eta{}.csv".format(lambda1,lambda2)
test_acc_file = "/home/wyx/vscode_projects/hier/models/CIFAR10/test_acc_lpsl_lambda{}eta{}.csv".format(lambda1,lambda2)
best_model_path = "/home/wyx/vscode_projects/hier/models/CIFAR10/best_model_lpsl_lambda{}eta{}.pth".format(lambda1,lambda2)
with open(train_loss_file, "w") as f:
    f.write("Epoch,Loss,Level1CE,Level2CE,Level3CE,LPS,HVP\n")
with open(val_loss_file, "w") as f:
    f.write("Epoch,Loss,Level1CE,Level2CE,Level3CE,LPS,HVP\n")
with open(test_acc_file, "w") as f:
    f.write("Epoch,Level1-ACC,Level2-ACC,Level3-ACC\n")

# 读取数据
dataset_train = CIFAR10Dataset(train=True, HLevel=HLevel, image_size=224)
dataset_test = CIFAR10Dataset(train=False, HLevel=HLevel, image_size=224)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,num_workers=16)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False,num_workers=16)

# loss 类型
criterion = LPSL(CIFAR10Dataset.CLASS_HIERARCHY,lambda1,lambda2)

model = models.vgg16(pretrained=True)
model.classifier = nn.Sequential(*[model.classifier[i] for i in range(6)])
model.classifier.add_module("logits", nn.Linear(4096, n_classes))
# 初始化 https://androidkt.com/initialize-weight-bias-pytorch/
model.classifier[-1].bias.data.fill_(-np.log(n_classes-1))
model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
            _, pred1 = torch.max(output.data[..., :2], dim=1)
            _, real1 = torch.max(target.data[..., :2], dim=1)
            _, pred2 = torch.max(output.data[..., 2:9], dim=1)
            _, real2 = torch.max(target.data[..., 2:9], dim=1)
            _, pred3 = torch.max(output.data[..., 9:], dim=1)
            _, real3 = torch.max(target.data[..., 9:], dim=1)
            
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
for epoch in range(c_epoch+1, EPOCHS + 1):
    print("Training....")
    train(epoch)
    print("Testing....")
    acc = val(epoch)
    if np.mean(acc) > best_acc:
        best_acc = np.mean(acc)
        torch.save(model.state_dict(), best_model_path)
    