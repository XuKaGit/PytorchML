import torch
from torch import nn
 
# 搭建神经网络（10分类网络）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 把网络放到序列中
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2), #输入是32x32的，输出还是32x32的（padding经计算为2）
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),  #输入输出都是16x16的（同理padding经计算为2）
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),   
            nn.Linear(in_features=64*4*4,out_features=64),
            nn.Linear(in_features=64,out_features=10)
        )
    
    def forward(self,x):
        x = self.layer(x)
        return x
 
if __name__ == '__main__':
    
    # 测试网络的验证正确性
    Net= Net()
    input = torch.ones((64,3,32,32))  # batch_size=64（代表64张图片）,3通道，32x32
    output = Net(input)
    print(output.shape)