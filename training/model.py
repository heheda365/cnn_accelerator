import torch
import torch.nn as nn
import torch.nn.functional as F
import quant_dorefa
import shiftCov


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,32,3,padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32,32,3,padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*14*14,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10)
        )
    def forward(self,x):
        in_size = x.size(0)
        out = self.features(x)
        out = out.view(in_size,-1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return out

class ShiftConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,1)
        self.shift1 = shiftCov.Shift3x3(16)  
        self.conv2 = nn.Conv2d(16,32,1)
        self.shift2 = shiftCov.Shift3x3(32)
        self.conv3 = nn.Conv2d(32,32,1) 
        self.shift3 = shiftCov.Shift3x3(32)
        self.conv4 = nn.Conv2d(32, 32, 1)
        self.fc1 = nn.Linear(32*14*14,128)
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x) # 28
        out = F.relu(out)
        out = self.shift1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.shift2(out)
        out = F.relu(out)         # 28
        out = F.max_pool2d(out, 2, 2)  #14
        out = self.conv3(out)   #14
        out = F.relu(out)       # 14
        out = self.shift3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out

class DWConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,1)
        # self.shift1 = shiftCov.Shift3x3(16)  
        self.dwconv1 = nn.Conv2d(16, 16, 3, padding=(1, 1), groups=16)
        self.conv2 = nn.Conv2d(16,32,1)
        # self.shift2 = shiftCov.Shift3x3(32)
        self.dwconv2 = nn.Conv2d(32, 32, 3, padding=(1, 1), groups=32)
        self.conv3 = nn.Conv2d(32,32,1) 
        # self.shift3 = shiftCov.Shift3x3(32)
        self.dwconv3 = nn.Conv2d(32, 32, 3, padding=(1, 1), groups=32)
        self.conv4 = nn.Conv2d(32, 32, 1)
        self.fc1 = nn.Linear(32*14*14,128)
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        in_size = x.size(0)
        # print('in_size = ', in_size)
        out = self.conv1(x) # 28
        out = F.relu(out)
        out = self.dwconv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.dwconv2(out)
        out = F.relu(out)         # 28
        out = F.max_pool2d(out, 2, 2)  #14
        out = self.conv3(out)   #14
        out = F.relu(out)       # 14
        out = self.dwconv3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = out.view(in_size,-1)
        # print(out.size())
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out

class QuantShiftConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        Conv2d = quant_dorefa.conv2d_Q_fn(w_bit=4)
        self.act_q = quant_dorefa.activation_quantize_fn(a_bit=4)
        Linear = quant_dorefa.linear_Q_fn(w_bit=4)
        # self.shift3x3 = shiftCov.Shift3x3()
        
        # 1,28x28
        # self.conv1 = nn.Conv2d(1,10,5, padding_mode='some') # 10, 24x24
        self.conv1 = Conv2d(1,16,1)
        self.shift1 = shiftCov.Shift3x3(16)  
        self.conv2 = Conv2d(16,32,1)
        self.shift2 = shiftCov.Shift3x3(32)
        self.conv3 = Conv2d(32,32,1) 
        self.shift3 = shiftCov.Shift3x3(32)
        self.conv4 = Conv2d(32, 32, 1)
        self.fc1 = Linear(32*14*14,128)
        self.fc2 = Linear(128,10)
    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x) # 28
        out = F.relu(out)
        out = self.act_q(out)
        out = self.shift1(out)
        out = F.relu(out)
        out = self.act_q(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.act_q(out)
        out = self.shift2(out)
        out = F.relu(out)         # 28
        out = self.act_q(out)
        out = F.max_pool2d(out, 2, 2)  #14
        out = self.conv3(out)   #14
        out = F.relu(out)       # 14
        out = self.shift3(out)
        out = F.relu(out)
        out = self.act_q(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = self.act_q(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.act_q(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out

class MiniConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32,32,3,padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32,32,3,padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32,32,3,padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*7*7,20),
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            nn.Linear(20,10)
        )
    def forward(self,x):
        in_size = x.size(0)
        out = self.features(x)
        out = out.view(in_size,-1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return out       

if __name__ == "__main__":

    import torchvision
    model = torchvision.models.alexnet(pretrained=False) #我们不下载预训练权重
    print(model)
    model = ConvNet()
    print(model)