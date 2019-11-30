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
#include "pool2d.h"
#include "bn_qrelu2d.h"


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
    // ap_uint<32> ap_test[32] = {0};
    // cout << "ap_test \n";
    // for(unsigned i=0; i < 32; i ++) {
    //     cout << ap_test[i] << "  ";
    // }
    // cout << "\n";
    // int q = bn_qurelu<32, CONV_0_OUT_BIT, CONV_0_INC_BIT, CONV_0_BIAS_BIT>(0, 34, 128);
    // cout << "q = " << q << "\n";
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
    hls::stream<ap_uint<CONV_0_OUT_BIT * CONV_0_OFM_CH>>  conv_0_out("conv_0_out");
    conv2d_bn_act<         CONV_0_K, 
                    CONV_0_S, 
                    CONV_0_P, 

                    CONV_0_IFM_ROW, 
                    CONV_0_IFM_COL, 
                    CONV_0_IFM_CH, 
                    CONV_0_IN_BIT, 

                    CONV_0_OFM_CH, 
                    CONV_0_OUT_BIT,

                    CONV_0_W_BIT, 
                    32,                     // 32 目前指乘累加结果的值的位数 应该用不到32位
                    CONV_0_INC_BIT,
                    CONV_0_BIAS_BIT,

                    CONV_0_SIMD, 
                    CONV_0_PE>(
                in_stream, 
                conv_0_w, 
                conv_0_inc,
                conv_0_bias,
                conv_0_out );

    // 接下来是 max_pool层
    hls::stream<ap_uint<CONV_0_OUT_BIT*CONV_0_OFM_CH>> pool_0_out("pool_out");
    max_pool2d< 2, 
                2, 
                CONV_0_OFM_ROW, 
                CONV_0_OFM_COL, 
                CONV_0_OFM_CH, 
                CONV_0_OUT_BIT>(
                    conv_0_out, 
                    pool_0_out);
    
// 未使用激活函数时 结果是正确的

    // hls::stream<ap_uint<4>> out("out");
    // adjust_width<32 * 4, 4, 14 *14>(pool_0_out, out);

    // cout << "pool out \n";
    // for(int i=0; i < 14;  i ++) {
    //     for(int j=0; j < 14; j ++) {
    //         for(int k=0; k < 32; k ++) {
    //             ap_uint<4> out_num = out.read();
    //             // if(k == 4)
    //             cout << out_num << "  "; 

    //         }
    //         cout << "\n";
    //     }
    //     cout << "\n";
    // }
    
// pool_0_out对比一致
    
    
    // cout << "conv_1================================\n";
    hls::stream<ap_uint<CONV_1_OUT_BIT * CONV_1_OFM_CH>>  conv_1_out("conv_1_out");
    conv2d_bn_act<  CONV_1_K, 
                    CONV_1_S, 
                    CONV_1_P, 

                    CONV_1_IFM_ROW, 
                    CONV_1_IFM_COL, 
                    CONV_1_IFM_CH, 
                    CONV_1_IN_BIT, 

                    CONV_1_OFM_CH, 
                    CONV_1_OUT_BIT,

                    CONV_1_W_BIT, 
                    32, 
                    CONV_1_INC_BIT,
                    CONV_1_BIAS_BIT,

                    CONV_1_SIMD, 
                    CONV_1_PE>(
                pool_0_out, 
                conv_1_w, 
                conv_1_inc,
                conv_1_bias,
                conv_1_out );
    
    // 单独的 bn_relu 层用于测试
    // hls::stream<ap_uint<CONV_1_OUT_BIT> > bn_relu_out;
    // bn_qrelu2d< CONV_1_IFM_ROW, 
    //             CONV_1_IFM_COL, 
    //             CONV_1_IFM_CH, 
    //             32, 

    //             CONV_1_OUT_BIT,
    //             CONV_1_INC_BIT,
    //             CONV_1_BIAS_BIT,
    //             CONV_1_PE> (
    //                 conv_1_out,
    //                 conv_1_inc,
    //                 conv_1_bias,
    //                 bn_relu_out
    //             );
                
    
    hls::stream<ap_uint<CONV_2_OUT_BIT * CONV_2_OFM_CH>>  conv_2_out("conv_2_out");
    conv2d_bn_act<  CONV_2_K, 
                    CONV_2_S, 
                    CONV_2_P, 

                    CONV_2_IFM_ROW, 
                    CONV_2_IFM_COL, 
                    CONV_2_IFM_CH, 
                    CONV_2_IN_BIT, 

                    CONV_2_OFM_CH, 
                    CONV_2_OUT_BIT,

                    CONV_2_W_BIT, 
                    32, 
                    CONV_2_INC_BIT,
                    CONV_2_BIAS_BIT,

                    CONV_2_SIMD, 
                    CONV_2_PE>(
                conv_1_out, 
                conv_2_w, 
                conv_2_inc,
                conv_2_bias,
                conv_2_out );
    
    // 第二个 max_pool层
    hls::stream<ap_uint<CONV_2_OUT_BIT*CONV_2_OFM_CH>> pool_1_out("pool_out");
    max_pool2d< 2, 
                2, 
                CONV_2_OFM_ROW, 
                CONV_2_OFM_COL, 
                CONV_2_OFM_CH, 
                CONV_2_OUT_BIT>(
                    conv_2_out, 
                    pool_1_out);


    // 卷积层
    hls::stream<ap_uint<CONV_3_OUT_BIT * CONV_3_OFM_CH>>  conv_3_out("conv_3_out");
    conv2d_bn_act<  CONV_3_K, 
                    CONV_3_S, 
                    CONV_3_P, 

                    CONV_3_IFM_ROW, 
                    CONV_3_IFM_COL, 
                    CONV_3_IFM_CH, 
                    CONV_3_IN_BIT, 

                    CONV_3_OFM_CH, 
                    CONV_3_OUT_BIT,

                    CONV_3_W_BIT, 
                    32, 
                    CONV_3_INC_BIT,
                    CONV_3_BIAS_BIT,

                    CONV_3_SIMD, 
                    CONV_3_PE>(
                pool_1_out, 
                conv_3_w, 
                conv_3_inc,
                conv_3_bias,
                conv_3_out );
    
    // 四个卷积层测试全部通过



    hls::stream<ap_uint<4>> out("out");
    adjust_width<32 * 4, 4, 7 *7>(conv_3_out, out);

    cout << "conv_2_out \n";
    for(int i=0; i < 7;  i ++) {
        for(int j=0; j < 7; j ++) {
            for(int k=0; k < 32; k ++) {
                ap_uint<4> out_num = out.read();
                // if(k == 0) 
                    cout << out_num << "  "; 
            }
            cout << "\n";

        }
        cout << "\n";
    }
    ap_uint<4> test_q = bn_qurelu<32, 4, CONV_1_INC_BIT, CONV_1_BIAS_BIT>(-8, 0x1c, 0x62);
    cout << "test_q = " << test_q;

    // test w
    // int res = 0;
    // cout << "conv_1_w  \n";
    // for(int i=0; i < 2 * 9; i ++) {
    //     for(int j=0; j < 16; j ++) {
    //         ap_int<2> w = conv_1_w[0][i](2*(j+1)-1, 2*j);
    //         res += w * out.read();
    //         cout << res << "  ";
    //     }
    //     cout << "\n";
    // }
    
    // ap_uint<4> a = 10;
    // ap_uint<4> b = 10;
    // ap_uint<4> c = a + b;

    // for(int i=0; i < CONV_0_OFM_CH; i ++) {
    //     cout << "inc" << 
    // }

    // cout << "c = " << c << "\n";
    // // 会直接溢出
    // for(int i=-200; i < 400; i ++) {
    //     int o = bn_qurelu<32, CONV_0_OUT_BIT, CONV_0_INC_BIT, CONV_0_BIAS_BIT>(i, 2, -7);
    //     cout << i << "\t" << o << "\n";
    // }



    return 0;
}

// 0  1  2  3  4  5  6  7  8   9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29  30 31
// 0  6  0  0  0  1  5  0  11  1  6  6  0  0  1  7  0  0  5  0  0  1  0  6  0  0  7  0  1  11  6  9
// 0  0  0  0  0  0  0  0  3   1  6  0  0  0  0  7  0  0  5  0  0  1  0  0  0  0  2  0  1   1  6  9 
//    1           5  6     7         8        11                         23       26        29                         



// 0  
// 0  
// 0  
// 0  
// 0  
// 0  
// 0  
// 0  
// 0  
// 0  
// -13  
// -27  
// -27  
// -27  
// -27  
// -47  
// -53  
// -70 


// -70  32722  -1156099061  32743  -1156099894  32715  62  -31  -1156099945  32781  4548238  -12  -1156099930  32760  4533254  11  -5  12  -24  10  -34  39  -62  57  -1  -15  0  9  7  -30  -1  15  


// -70  -28  4  -22  10  -20  13  -16  7  1  -43  3  -11  10  14  26  -5  -3  -39  10  -19  24  -77  57  -1  15  0  -6  7  -30  -1  15  
// -69  -31  -19  -33  29  -28  30  -19  -8  36  -38  -1  -19  10  29  46  -7  0  -22  20  -14  34  -87  56  -20  28  16  -3  -8  -7  -8  27  
// -69  -31  -19  -33  29  -28  30  -19  -8  36  -38  -1  -19  10  29  46  -7  0  -22  20  -14  34  -87  56  -20  28  16  -3  -8  -7  -8  27


// 0  0  1  0  0  0  0  0  0  0  0  6  3  3  0  5  3  0  0  0  0  0  0  0  15  0  6  0  0  0  8  4  
// 0  0  0  0  1  0  3  0  0  4  0  5  2  3  0  7  2  0  0  0  0  0  0  0  15  0  8  0  0  0  7  5  
// 0  0  0  0  1  0  3  0  0  4  0  5  2  3  0  7  2  0  0  0  0  0  0  0  15  0  8  0  0  0  7  5 

// 下面是正确的
// 0  0  1  0  0  0  0  0  0  0  0  6  3  3  0  5  3  0  0  0  0  0  0  0  3  0  6  0  0  0  3  4  
// 0  0  0  0  1  0  0  0  0  4  0  5  2  3  0  7  2  0  0  0  0  0  0  0  2  0  8  0  0  0  3  5  
// 0  0  0  0  1  0  0  0  0  4  0  5  2  3  0  7  2  0  0  0  0  0  0  0  2  0  8  0  0  0  3  5 
                                                                           