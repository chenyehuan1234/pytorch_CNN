import time
import copy
import pandas as pd
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet



def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    train_data,val_data=Data.random_split(train_data,[int(len(train_data)*0.8),int(len(train_data)*0.2)])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=0  # Windows 修复
                                       )

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=0  # Windows 修复
                                       )

    return train_dataloader,val_dataloader



def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #优化器,学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    #损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入训练设备中
    model = model.to(device)
    #复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    #训练集，验证集损失列表
    train_loss_all = []
    val_loss_all = []

    # 训练集，验证集准确度列表
    train_acc_all = []
    val_acc_all = []
    #当前时间
    since= time.time()


    for epoch in range(num_epochs):
        print("Epoch{}/{}".format(epoch,num_epochs-1))
        print("-"*50)

        #初始化参数
        #训练集损失函数，准确度
        train_loss = 0.0
        train_corrects = 0

        # 验证集损失函数，准确度
        val_loss = 0.0
        val_corrects = 0

        #训练集和验证集样本数量
        train_num = 0
        val_num = 0

        #对每一个mini_batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):  # 修复：train_loader → train_dataloader
            #将特征，标签放到训练设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #训练模式
            model.train()

            #前向传播过程，输入为一个batch，输出位一个batch对应的预测
            output = model(b_x)

            #查找每一行中最大值对应的行标
            pre_lab= torch.argmax(output,dim=1)

            #计算每一个batch的损失函数
            loss = criterion(output,b_y)



            #将梯度初始化为0
            optimizer.zero_grad()

            #反向传播计算
            loss.backward()

            #根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()

            #对损失函数进行累加
            train_loss+=loss.item() * b_x.size(0)

            #如果预测正确，则准确度train_corrects 加1
            train_corrects += torch.sum(pre_lab == b_y.data)

            #当前用于训练的样本数量
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            #将特征，标签放到验证设备
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #设置模型为评估模式
            model.eval()

            # 验证集不需要计算梯度，修复显存
            with torch.no_grad():
                #前向传播过程，输入为一个batch，输出位一个batch对应的预测
                output = model(b_x)

                #查找每一行中最大值对应的行标
                pre_lab= torch.argmax(output,dim=1)

                #计算每一个batch的损失函数，不需要反向传播
                loss = criterion(output,b_y)

                #对损失函数进行累加
                val_loss+=loss.item() * b_x.size(0)

                #如果预测正确，则准确度train_corrects 加1
                val_corrects += torch.sum(pre_lab == b_y.data)

                #当前用于验证的样本数量
                val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        val_loss_all.append(val_loss / val_num)  # 修复：必须除以总数
        val_acc_all.append(val_corrects.double().item() / val_num)

        print('{} Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} Val Loss:{:.4f} Val Acc:{:.4f}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))

        #寻找最高准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]

            best_model_wts = copy.deepcopy(model.state_dict())

        #训练耗费时间
        time_use=time.time()-since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))


    #选择最优参数
    #加载最高准确率下的模型参数
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),"./model.pth")  # 修复：保存模型

    train_process = pd.DataFrame(data={'epoch':range(num_epochs),  # 修复：小写匹配画图
                                       'train_loss_all':train_loss_all,
                                       'val_loss_all': val_loss_all,
                                       'train_acc_all': train_acc_all,
                                       'val_acc_all': val_acc_all

                                       })
    return train_process



def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,14))
    plt.subplot(1,2,1)
    plt.plot(train_process['epoch'],train_process.train_loss_all,'ro-',label='Train Loss')
    plt.plot(train_process['epoch'], train_process.val_loss_all, 'bs-', label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1,2,2)
    plt.plot(train_process['epoch'],train_process.train_acc_all,'ro-',label='Train Acc')
    plt.plot(train_process['epoch'], train_process.val_acc_all, 'bs-', label='Val Acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Acc')

    plt.show()

if __name__ == '__main__':
    #模型实例化
    LeNet = LeNet()
    train_dataloader,val_dataloader=train_val_data_process()
    train_process = train_model_process(LeNet, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)