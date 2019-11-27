#include <iostream>
#include <fstream>
#include <stdint.h>
#include <ap_int.h>
#include "stream_tools.h"
#include "debug.h"
#include "mnist-cnn-config-1W5A.h"
#include "mnist-cnn-params-1W5A.h"
#include "function.h"
#include "sliding_window_unit.h"
#include "matrix_vector_unit.h"
#include "config.h"
#include "param.h"
#include "conv2d.h"


void load_data(const char *path, char *ptr, unsigned int size)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f)
    {
        std::cout << "no such file,please check the file name!/n";
        exit(0);
    }
    f.read(ptr, size);
    f.close();
}

int main(int argc, char const *argv[])
{
    float data_raw[28][28];
    load_data("data/test_data.bin", (char *)data_raw, sizeof(data_raw));

    // 输入数据c仿真时两种写法都可以
    uint8_t data[28][28];
    hls::stream<ap_uint<8>> in_stream("in_stream");
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            data[i][j] = (int)(data_raw[i][j] * 255);
            in_stream.write(data[i][j]);
            std::cout << (int)data[i][j] << "  ";
        }
        std::cout << "\n";
    }

    // monitor 功能有待完善
    //  monitor<L0_Din, L0_Cin, L0_Ibit>(in_stream, "./log/mon_in_stream.log", 1);
    
    // padding 测试通过
    // hls::stream<ap_uint<8>> padding_stream("padding_stream");
    // padding<L0_Din, L0_Din, L0_Cin, L0_Ibit, 1>(in_stream, padding_stream);

    // // for (int i = 0; i < 30; i++)
    // // {
    // //     for (int j = 0; j < 30; j++)
    // //     {
    // //         std::cout << padding_stream.read() << "  ";
    // //     }
    // //     std::cout << "\n";
    // // }
    // // cout << padding_stream.size() << "===\n";

    // hls::stream<ap_uint<8>> swu_stream("padding_stream");
    // SWU<L0_K, L0_S, L0_Din + 2, L0_Din + 2, L0_Cin, L0_Ibit>(padding_stream, swu_stream);
    // // 处理得到输入矩阵

    // // stream<ap_uint<L0_MVTU_InP*L0_Ibit> > swu_out_reduced("swu_out_reduced");

    // // 将数据位宽调整到 1 × 8
	// // ReduceWidth<L0_Cin*L0_Ibit, L0_MVTU_InP*L0_Ibit, L0_K*L0_K*L0_Din*L0_Din> (swu_stream, swu_out_reduced, 1);

    // // 将数据位宽调整到 9 × 8
    // stream<ap_uint<9*8> > swu_out_expand("swu_out_expand");
    // ExpandWidth<8, 9 * 8, L0_K*L0_K*L0_Din*L0_Din / 9>(swu_stream, swu_out_expand, 1);

    // cout << "swu_out_expand.size = " << swu_out_expand.size() << "\n";


	// stream<ap_uint<4 * L0_Mbit> > out_raw("out_raw");

    
    // MVU<L0_Din * L0_Din, L0_Ibit, 2, L0_Mbit, L0_Cin * L0_K * L0_K, L0_Cout, 9, 4>(swu_out_expand, conv_0_w, out_raw);

    // cout << "out_raw.size = " << out_raw.size() << "==\n";

    // // 将计算得到的结果 维度降低
    // hls::stream<ap_uint<L0_Mbit>> out("mvu out");
    hls::stream<ap_uint<32 * CONV_0_OFM_CH>>  conv_0_out("conv_0_out");
    conv2d_bn_act<  CONV_0_K, 
                    CONV_0_S, 
                    CONV_0_P, 

                    CONV_0_IFM_ROW, 
                    CONV_0_IFM_COL, 
                    CONV_0_IFM_CH, 
                    CONV_0_IN_BIT, 

                    CONV_0_OFM_CH, 
                    CONV_0_OUT_BIT,

                    CONV_0_W_BIT, 
                    32, 
                    CONV_0_INC_BIT,
                    CONV_0_BIAS_BIT,

                    CONV_0_SIMD, 
                    CONV_0_PE>(
                in_stream, 
                conv_0_w, 
                conv_0_inc,
                conv_0_bias,
                conv_0_out );



    hls::stream<ap_uint<32>> out("out");
    adjust_width<32 * 32, 32, 28 *28>(conv_0_out, out);

    cout << "out \n";
    for(int i=0; i < 28;  i ++) {
        for(int j=0; j < 28; j ++) {
            for(int k=0; k < 32; k ++) {
                ap_int<32> out_num = out.read();
                if(k == 2) {
                    cout << out_num << "  "; 
                }
            }
        }
        cout << "\n";
    }
    
    ap_uint<4> a = 10;
    ap_uint<4> b = 10;
    ap_uint<4> c = a + b;
    cout << "c = " << c << "\n";
    // 会直接溢出




    return 0;
}
