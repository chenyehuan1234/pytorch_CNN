import torch.utils.data as Data
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import LeNet


def test_val_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0  # Windows 修复
                                       )

    return test_dataloader

def test_model_process(model,test_dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    test_corrects = 0.0
    test_num = 0

    #只进行前向传播计算，不计算梯度，节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            #将特征和标签放入到测试设备中
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()

            #前向传播过程中，输入为测试数据集，输出为对每个样本的预测值
            output = model(test_data_x)
            pre_lab = output.argmax(dim=1)

            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_y.size(0)

    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：",test_acc)


if __name__ == "__main__":
    model = LeNet()
    model.load_state_dict(torch.load("./model/LeNet.pth"))
