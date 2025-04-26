import torchvision.datasets
import torchvision.transforms as transforms
from model import *
from torch import nn
from torch.utils.data import DataLoader
 
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])


mnist_train = torchvision.datasets.MNIST(
    root="../Data/MNIST", train=True, transform=transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root="../Data/MNIST", train=False, transform=transform, download=True)

# 加载测试集
train_loader = DataLoader(dataset=mnist_train,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
test_loader = DataLoader(dataset=mnist_test,batch_size=64,shuffle=True,num_workers=0,drop_last=True)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 创建网络模型
Net = Net()
Net = Net.to(device)
 
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()   # 分类问题可以用交叉熵
loss_fn = loss_fn.to(device)
 
# 定义优化器
learning_rate = 0.01   # 另一写法：1e-2，即1x 10^(-2)=0.01
optimizer = torch.optim.SGD(Net.parameters(),lr=learning_rate)   # SGD 随机梯度下降
 
# 设置训练网络的一些参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 2   # 训练轮数
 
for i in range(epoch):
    print("----------第{}轮训练开始-----------".format(i+1))  # i从0-9
    # 训练步骤开始
    Net.train()  # 训练模式
    for data in train_loader:   # 从训练的dataloader中取数据
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = Net(imgs)
        loss = loss_fn(outputs,targets)
 
        # 优化器优化模型
        optimizer.zero_grad()    # 首先要梯度清零
        loss.backward()  # 反向传播得到每一个参数节点的梯度
        optimizer.step()   # 对参数进行优化
        total_train_step += 1
        print("训练次数：{},loss:{}".format(total_train_step,loss.item()))

    # 测试步骤开始
    total_test_loss = 0
    Net.eval()  # 测试模式
    with torch.no_grad():  # 无梯度，不进行调优
        for data in test_loader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = Net(imgs)
            loss = loss_fn(outputs,targets)  # 该loss为部分数据在网络模型上的损失，为tensor数据类型
            # 求整体测试数据集上的误差或正确率
            total_test_loss = total_test_loss + loss.item()  # loss为tensor数据类型，而total_test_loss为普通数字
    print("整体测试集上的Loss:{}".format(total_test_loss))
    
    #torch.save(Net.state_dict(),"Net_{}.pth".format(i))  # 每一轮保存一个结果, ./Net_{}.pth
    #   model = torch.load("Net_1.pth")
    # 使用gpu训练保存的模型在cpu上使用: model = torch.load("XXXX.pth",map_location= torch.device("cpu"))
    #print("模型已保存")