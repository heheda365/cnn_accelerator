from qnn_param_reader import QNNParamReader
from qnn_mem_process import QNNLayerMemProcess
import json
import os
import sys

    


if __name__ == "__main__":

    target_dir_hls_param = 'param/hls/'
    if not os.path.exists(target_dir_hls_param):
        os.makedirs(target_dir_hls_param)
    
    hls_param_file = open(target_dir_hls_param + 'param.h', 'w')
    hls_config_file = open(target_dir_hls_param + 'config.h', 'w')

    config_file = open('config.json', 'r', encoding='utf-8')
    config = json.load(config_file)
    reader = QNNParamReader('miniConvNet.npz')

    # conv_0
    processer = QNNLayerMemProcess('conv_0', reader, config, w_bit=2, in_bit=8, out_bit=4, l_shift=0, pe=4, simd=9)
    w, inc, bias = processer.conv()
    param_str = processer.layer_param_to_init_str(w, inc, bias)
    config_str = processer.conv_config_str()
    hls_param_file.write(param_str)
    hls_config_file.write(config_str)

    # conv_1
    processer = QNNLayerMemProcess('conv_1', reader, config, w_bit=2, in_bit=4, out_bit=4, l_shift=0, pe=16, simd=16)
    w, inc, bias = processer.conv()
    param_str = processer.layer_param_to_init_str(w, inc, bias)
    config_str = processer.conv_config_str()
    hls_param_file.write(param_str)
    hls_config_file.write(config_str)

    # conv_2
    processer = QNNLayerMemProcess('conv_2', reader, config, w_bit=2, in_bit=4, out_bit=4, l_shift=0, pe=16, simd=16)
    w, inc, bias = processer.conv()
    param_str = processer.layer_param_to_init_str(w, inc, bias)
    config_str = processer.conv_config_str()
    hls_param_file.write(param_str)
    hls_config_file.write(config_str)

    # conv_3
    processer = QNNLayerMemProcess('conv_3', reader, config, w_bit=2, in_bit=4, out_bit=4, l_shift=0, pe=16, simd=16)
    w, inc, bias = processer.conv()
    param_str = processer.layer_param_to_init_str(w, inc, bias)
    config_str = processer.conv_config_str()
    hls_param_file.write(param_str)
    hls_config_file.write(config_str)

    # linear_0
    processer = QNNLayerMemProcess('linear_0', reader, config, w_bit=2, in_bit=4, out_bit=4, l_shift=0, pe=4, simd=16)
    w, inc, bias = processer.linear()
    param_str = processer.layer_param_to_init_str(w, inc, bias)
    config_str = processer.linear_config_str()
    hls_param_file.write(param_str)
    hls_config_file.write(config_str)

    # linear_1  last linear

    hls_param_file.close()
    hls_config_file.close()