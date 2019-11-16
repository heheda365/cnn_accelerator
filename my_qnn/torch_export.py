import torch    
import model
import numpy as np

# 小型量化网络
model = model.MiniConvNet()
model.load_state_dict(torch.load('model.pkl', map_location='cpu'))

def generate_params(model):
    dic = {}
    cnt = 0
    for sub_module in model.modules():
        if type(sub_module).__base__ is torch.nn.Conv2d:
            w = sub_module.weight.detach().numpy()
            dic['arr_' + str(cnt)] = w
            cnt = cnt + 1
        elif type(sub_module).__base__ is torch.nn.Linear:
            w = sub_module.weight.detach().numpy()
            dic['arr_' + str(cnt)] = w
            cnt = cnt + 1
        elif type(sub_module) is torch.nn.BatchNorm2d or type(sub_module) is torch.nn.BatchNorm1d:
            gamma = sub_module.weight.detach().numpy()
            dic['arr_' + str(cnt)] = gamma
            cnt = cnt + 1
            beta = sub_module.bias.detach().numpy()
            dic['arr_' + str(cnt)] = beta
            cnt = cnt + 1
            mean = sub_module.running_mean.numpy()
            dic['arr_' + str(cnt)] = mean
            cnt = cnt + 1
            var = sub_module.running_var.numpy()
            dic['arr_' + str(cnt)] = var
            cnt = cnt + 1
            eps = sub_module.eps
            dic['arr_' + str(cnt)] = eps
            cnt = cnt + 1
    return dic

dic = generate_params(model)
np.savez('miniConvNet.npz', **dic)
            