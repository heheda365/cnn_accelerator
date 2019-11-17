import torch    
import model
import numpy as np

model = model.MiniConvNetFull()
model.load_state_dict(torch.load('model.pkl', map_location='cpu'))

def generate_params(model):
    dic = {}
    cnt = 0
    for sub_module in model.named_modules():
        if type(sub_module) is torch.nn.Conv2d:
            w = sub_module.weight.detach().numpy()
            dic['arr_' + cnt] = w
            cnt = cnt + 1
        elif type(sub_module) is torch.nn.Linear:
            w = sub_module.weight.detach().numpy()
            dic['arr_' + cnt] = w
            cnt = cnt + 1
        elif type(sub_module) is torch.nn.BatchNorm2d or type(sub_module) is torch.nn.BatchNorm1d:
            gamma = sub_module.weight.detach().numpy()
            dic['arr_' + cnt] = gamma
            cnt = cnt + 1
            beta = sub_module.bias.detach().numpy()
            dic['arr_' + cnt] = beta
            cnt = cnt + 1
            mean = sub_module.running_mean.numpy()
            dic['arr_' + cnt] = mean
            cnt = cnt + 1
            var = sub_module.running_var.numpy()
            dic['arr_' + cnt] = var
            cnt = cnt + 1
            eps = sub_module.eps
            dic['arr_' + cnt] = eps
            cnt = cnt + 1
    return dic

dic = generate_params(model)
np.savez('MiniConvNet.npz', **dic)
            