from qnn_param_reader import QNNParamReader
import numpy as np
import os
import sys

# 将数组中元素拼接组合
# 例如 输入 [1, 1, 1] elem_bit = 1, 返回 111
# 返回值是 int 类型 其程度可能超过 64位
def array_to_string(array, elem_bit):
        val = 0	
        #for i in range(len(array)-1, -1, -1):
        for i in range(len(array)):
            tmp = array[i]
            tmp2 = tmp

            if tmp < 0:
                tmp2 = 2**(elem_bit) + tmp 

            tmp2 = int(tmp2)
            tmp3 = tmp2 * 2**(elem_bit*i)
            val = val + tmp3
        return val


#  处理得到 转化后的w矩阵和 bn的inc，bias
# w矩阵为二维矩阵，row为输出通道数
# 相对于原始w张量，其内存排列顺序转换为 out_ch, row, col, in_ch
class ParamProcess:
    def __init__(self, file_name):
        self.qnn_read = QNNParamReader(file_name)

    def conv_process(self, w_bit, in_bit, out_bit, l_shift):
        con_w = self.qnn_read.read_qconv_weight(w_bit)
        # con_w 是一个4维张量
        # 将输入通道维度放到最后
        con_w.transpose(0, 2, 3, 1)
        # 处理为二维矩阵
        con_w = con_w.reshape(con_w.shape[0], -1)

        # qinc, qbias 当前不需要处理
        qinc, qbias = self.qnn_read.read_qbarch_norm_act_param(w_bit, in_bit, out_bit, l_shift)

        return con_w, qinc, qbias
    
    def linear_process(self, w_bit, in_bit, out_bit, l_shift):
        # linear_w0 是一个二维矩阵不需要处理
        linear_w0 = self.qnn_read.read_qlinear_weight(w_bit)
        linear_bn0_inc, linear_bn0_bias = self.qnn_read.read_qbarch_norm_act_param(w_bit, in_bit, out_bit, l_shift)

        return linear_w0, linear_bn0_inc, linear_bn0_bias

    def last_linear_process(self, w_bit):
        # 全连接层
        linear_w0 = self.qnn_read.read_qlinear_weight(w_bit=2)
        return linear_w0

# 将参数整理成满足 硬件设计需求的形式
class QNNMemProcess:
    def __init__(self, param_file_name):
        self.param_process = ParamProcess(param_file_name)

    # 将矩阵整理成所需要的储存样式
    # 转化位 pe * titles 矩阵
    def w_to_hls_array(self, w, w_bit, simd, pe):
        assert w.shape[0] % pe == 0, 'out_ch mod pe must 0'
        # w 矩阵的宽 其值 为 k * k * in_ch
        h = w.shape[1]
        # res0 size = out_ch, k * k * in_ch // simd + (0 or 1)
        res0 = [[0 for i in range(h // simd)] for j in range(w.shape[0])]
        for out_ch in range(w.shape[0]):
            for i in range(h // simd):
                arr = w[out_ch][i*simd:(i+1)*simd]
                res0[out_ch][i] = array_to_string(arr, w_bit)
            
        # 处理不够整除的部分
        if h % simd != 0:
            print('h mod simd != 0')
            for out_ch in range(w.shape[0]):
                arr = w[out_ch][h // simd * simd:]
                res0[out_ch].append(array_to_string(arr, w_bit))

        # print('res0 = ', len(res0), len(res0[0]))
        # print(np.array(res0))
        
        tiles = len(res0[0]) * (len(res0) // pe) 
        # print('tiles', tiles)
        res = [[0 for i in range(tiles)] for i in range(pe)]

        tiles_cnt = 0
        for i in range(len(res0) // pe):
            for j in range(len(res0[0])):

                for pe_cnt in range(pe):
                    res[pe_cnt][tiles_cnt] = res0[i * pe + pe_cnt][j]
                tiles_cnt += 1  
        return res

    # 处理 inc 和 bias
    def inc_bias_to_hls_array(self, inc, bias, pe):
        inc = inc.reshape(pe, -1)
        bias = bias.reshape(pe, -1)
        
        return inc, bias
    
    # 卷积参数整理
    # 返回的w因为元素可能大于64位 所以用list储存
    # inc, bias 是numpy.array类型
    def conv(self, w_bit, in_bit, out_bit, l_shift, pe, simd):
        # w 是二维矩阵形式
        w, inc, bias = self.param_process.conv_process(w_bit, in_bit, out_bit, l_shift)
        # print(w)
        # 先把 w 处理为每个元素位宽都是 simd * w_bit 形式
        w = self.w_to_hls_array(w, w_bit, simd, pe)

        inc, bias = self.inc_bias_to_hls_array(inc, bias, pe)

        return w, inc, bias



    def linear(self, w_bit, in_bit, out_bit, l_shift, pe, simd):
        w, inc, bias = self.param_process.linear_process(w_bit, in_bit, out_bit, l_shift)
        w = self.w_to_hls_array(w, w_bit, simd, pe)
        inc, bias = self.inc_bias_to_hls_array(inc, bias, pe)

        return w, inc, bias
    

    # 最后一个全连接层
    def last_linear(self, w_bit, pe, simd):
        w = self.param_process.last_linear_process(w_bit)
        w = self.w_to_hls_array(w, w_bit, simd, pe)
        return w
    
    def w_to_hls_init_str(self, name, w, w_bit, pe, simd) -> str:
        w_mem_type = "ap_uint<"+str(w_bit * simd)+">"

        res = '// ' + name + '\n'
        res += "//PEs = %d, SIMD width = %d\n" % (pe, simd)
        res += '//w_bit = %d\n' % w_bit
        res += w_mem_type
        res += (' ' + name) 
        res += '[%d][%d] = {\n' % (len(w), len(w[0]))

        res += ",\n".join(map(lambda pe:"{"+(", ".join(map(hex, pe)))+"}", w))
        res += '};\n'

        return res
    

    # 确定 inc 位宽 
    # 实验中发现inc 都为正数
    def get_inc_bit_width(self, inc):
        max_num = inc.max()
        bit_width = len(str(bin(max_num))) - 2
        return bit_width
    
    # 确定bias的位宽
    # bias 有整数和负数
    # 当前算法得出的还不是最优
    def get_bias_bit_width(self, bias):
        abs_max = np.abs(bias).max()
        bit_width = len(str(bin(abs_max))) - 2
        return bit_width + 1
    
    def inc_to_hls_init_str(self, name, inc, pe, simd) -> str:
        inc_bit_width = self.get_inc_bit_width(inc)

        w_mem_type = "ap_uint<"+str(inc_bit_width)+">"

        res = '// inc\n'
        res += '// ' + name + '\n'
        res += '//w_bit = %d\n' % inc_bit_width
        res += w_mem_type
        res += (' ' + name) 
        res += '[%d][%d] = {\n' % (len(inc), len(inc[0]))

        res += ",\n".join(map(lambda pe:"{"+(", ".join(map(hex, pe)))+"}", inc))
        res += '};\n'

        return res  
    
    def bias_to_hls_init_str(self, name, bias, pe, simd) -> str:
        bias_bit_width = self.get_bias_bit_width(bias)

        w_mem_type = "ap_int<"+str(bias_bit_width)+">"
        res = '// bias\n'
        res += '// ' + name + '\n'
        res += '//w_bit = %d\n' % bias_bit_width
        res += w_mem_type
        res += (' ' + name) 
        res += '[%d][%d] = {\n' % (len(bias), len(bias[0]))

        res += ",\n".join(map(lambda pe:"{"+(", ".join(map(hex, pe)))+"}", bias))
        res += '};\n'

        return res


    
    


        







if __name__ == "__main__":
    qnn_men_pro = QNNMemProcess('miniConvNet.npz')
    
    w, inc, bias = qnn_men_pro.conv(2, 8, 4, l_shift=0, pe=4, simd=9)
    w_str = qnn_men_pro.w_to_hls_init_str('conv_0_w', w, 2, 8, 9)
    inc_str = qnn_men_pro.inc_to_hls_init_str('conv_inc_0', inc, 8, 9)
    bias_str = qnn_men_pro.bias_to_hls_init_str('conv_bias_0', bias, 8, 9)

    print(w_str + inc_str + bias_str)

    # print(inc)

    # w, inc, bias = qnn_men_pro.conv(2, 4, 4, l_shift=0, pe=4, simd=32)
    # print(inc)
    # w, inc, bias = qnn_men_pro.conv(2, 4, 4, l_shift=0, pe=4, simd=32)
    # print(inc)

    # w, inc, bias = qnn_men_pro.conv(2, 4, 4, l_shift=0, pe=4, simd=32)
    # print(inc)

    # w, inc, bias = qnn_men_pro.linear(2, 4, 4, l_shift=0, pe=4, simd=32)
    # print(inc)
    # w_str = qnn_men_pro.w_to_hls_init_str('conv_0_w', w, 2, 4, 9)
    # print(np.array(w))
    # print(inc)
    # print(bias)
    # print(w_str)

    # a = -2
    # print(bin(a))
    # print(len(str(bin(a))))

    

    # # 维度顺序变换
    # a = [[[1, 2, 3], 
    #     [4, 5, 6],
    #     [7, 8, 9]],

    #     [[1, 2, 3], 
    #     [4, 5, 6],
    #     [7, 8, 9]],

    #     [[1, 2, 3], 
    #     [4, 5, 6],
    #     [7, 8, 9]],

    #     [[1, 2, 3], 
    #     [4, 5, 6],
    #     [7, 8, 9]]]
    # a = np.array(a)
    # print(a.shape[1])
    # b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # c = b[1:]
    # print(c)
    # print(a)
    # print(a.transpose(1, 2, 0))

    # print(a.reshape(4, -1))

    # def ArrayToString(array, precision, precFract=0, debug=False):
    #     val = 0	
    #     #for i in range(len(array)-1, -1, -1):
    #     for i in range(len(array)):
    #         tmp = array[i]
    #         tmp2 = tmp

    #         if tmp < 0:
    #             tmp2 = 2**(precision) + tmp 

    #         tmp2 = int(tmp2)
    #         tmp3 = tmp2 * 2**(precision*i)
    #         val = val + tmp3

    #     return val
    
    # a = ArrayToString([5, 6, 7, 8, 9, 1, 2, 2, 1, 2, 3, 4, 4], 40)
    # print(hex(a))

    # a = [1, 2]
    # b = a[0]   # 1
    # # b = 6
    # a[0] = 10
    # print(a[0], b)

    # a = [[1, 2], [3, 4]]
    # b = a[0]
    # a[0][0] = 10

    # a = 1
    # print(id(a))
    # a = a + 1
    # print(id(a))
    # print(b)

    # a = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
    # print(len(str(a)))

    # a = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]
    # b = ["%s " % str(x) for x in a]
    # c = [y for y in a]
    # print(b)
    # print(c)

    # def square(x):
    #     return x + 1
    # d = map(lambda pe: map(square, pe), a)  
    # # print(list())

    # e = '\n{'.join(map(lambda pe: ','.join(map(hex, pe)), a))
    # print(e)

    # f = ",\n".join(map(lambda pe:"{"+(", ".join(map(hex, pe)))+"}", a))
    # print(f)

    # print('a'.join())

    # a = []
    # a[0] = 1