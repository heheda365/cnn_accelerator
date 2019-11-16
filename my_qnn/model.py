import torch
import torch.nn as nn
import torch.nn.functional as F
import quant_dorefa


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        Conv2d = quant_dorefa.conv2d_Q_fn(w_bit=1)
        self.act_q = quant_dorefa.activation_quantize_fn(a_bit=32)
        Linear = quant_dorefa.linear_Q_fn(w_bit=1)
        
        # 1,28x28
        self.conv1=Conv2d(1,10,5, bias=True) # 10, 24x24
        # self.bn1=nn.BatchNorm2d(10)
        self.conv2=Conv2d(10,20,3, bias=True) # 128, 10x10
        # self.bn2=nn.BatchNorm2d(20)
        self.fc1 = Linear(20*10*10,50, bias=True)
        self.fc2 = Linear(50,10, bias=True)
    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x) #24
        # out = self.bn1(out)
        out = self.act_q(F.relu(out))
        out = F.max_pool2d(out, 2, 2)  #12
        out = self.conv2(out) #10
        # out = self.bn2(out)
        out = self.act_q(F.relu(out))
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = self.act_q(F.relu(out))
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out
class MiniConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        Conv2d = quant_dorefa.conv2d_Q_fn(w_bit=2)
        # self.act_q = quant_dorefa.activation_quantize_fn(a_bit=4)
        Linear = quant_dorefa.linear_Q_fn(w_bit=2)

        self.features = nn.Sequential(
            Conv2d(1,32,3,padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            quant_dorefa.activation_quantize_fn(a_bit=4),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d(32,32,3,padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # self.act_q(),
            quant_dorefa.activation_quantize_fn(a_bit=4),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d(32,32,3,padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # self.act_q(),
            quant_dorefa.activation_quantize_fn(a_bit=4),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d(32,32,3,padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # self.act_q(),
            quant_dorefa.activation_quantize_fn(a_bit=4)
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            Linear(32*7*7,20, bias=False),
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            # self.act_q(),
            quant_dorefa.activation_quantize_fn(a_bit=4),
            Linear(20,10, bias=False)
        )
    def forward(self,x):
        in_size = x.size(0)
        out = self.features(x)
        out = out.view(in_size,-1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return out  
class MiniConvNetFull(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv2d = quant_dorefa.conv2d_Q_fn(w_bit=1)
        # # self.act_q = quant_dorefa.activation_quantize_fn(a_bit=4)
        # Linear = quant_dorefa.linear_Q_fn(w_bit=1)

        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # quant_dorefa.activation_quantize_fn(a_bit=4),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32,32,3,padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # self.act_q(),
            # quant_dorefa.activation_quantize_fn(a_bit=4),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32,32,3,padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # self.act_q(),
            # quant_dorefa.activation_quantize_fn(a_bit=4),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32,32,3,padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # self.act_q(),
            # quant_dorefa.activation_quantize_fn(a_bit=4)
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*7*7,20, bias=False),
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            # self.act_q(),
            # quant_dorefa.activation_quantize_fn(a_bit=4),
            nn.Linear(20,10, bias=False)
        )
    def forward(self,x):
        in_size = x.size(0)
        out = self.features(x)
        out = out.view(in_size,-1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return out  


class MiniConvNetBNQ(nn.Module):
    def __init__(self):
        super().__init__()

        Conv2d = quant_dorefa.conv2d_Q_fn(w_bit=2)
        # self.act_q = quant_dorefa.activation_quantize_fn(a_bit=4)
        Linear = quant_dorefa.linear_Q_fn(w_bit=2)
        bn = quant_dorefa.batchNorm2d_Q_fn(w_bit=8)
        bn1 = quant_dorefa.batchNorm1d_Q_fn(w_bit=8)

        self.features = nn.Sequential(
            Conv2d(1,32,3,padding=(1, 1), bias=False),
            bn(32),
            nn.ReLU(inplace=True),
            quant_dorefa.activation_quantize_fn(a_bit=4),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d(32,32,3,padding=(1, 1), bias=False),
            bn(32),
            nn.ReLU(inplace=True),
            # self.act_q(),
            quant_dorefa.activation_quantize_fn(a_bit=4),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d(32,32,3,padding=(1, 1), bias=False),
            bn(32),
            nn.ReLU(inplace=True),
            # self.act_q(),
            quant_dorefa.activation_quantize_fn(a_bit=4),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d(32,32,3,padding=(1, 1), bias=False),
            bn(32),
            nn.ReLU(inplace=True),
            # self.act_q(),
            quant_dorefa.activation_quantize_fn(a_bit=4)
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            Linear(32*7*7,20, bias=False),
            # nn.BatchNorm1d(20),
            bn1(20),
            nn.ReLU(inplace=True),
            # self.act_q(),
            quant_dorefa.activation_quantize_fn(a_bit=4),
            Linear(20,10, bias=False)
        )
    def forward(self,x):
        in_size = x.size(0)
        out = self.features(x)
        out = out.view(in_size,-1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return out  

if __name__ == "__main__":
    model = MiniConvNetBNQ()
    for name, pra in model.named_parameters():
        print(name)
        # print(pra.size())