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
    hls::stream<ap_uint<8>> padding_stream("padding_stream");
    padding<L0_Din, L0_Din, L0_Cin, L0_Ibit, 1>(in_stream, padding_stream);

    // for (int i = 0; i < 30; i++)
    // {
    //     for (int j = 0; j < 30; j++)
    //     {
    //         std::cout << padding_stream.read() << "  ";
    //     }
    //     std::cout << "\n";
    // }
    // cout << padding_stream.size() << "===\n";

    hls::stream<ap_uint<8>> swu_stream("padding_stream");
    SWU<L0_K, L0_S, L0_Din + 2, L0_Din + 2, L0_Cin, L0_Ibit>(padding_stream, swu_stream);
    // 处理得到输入矩阵

    stream<ap_uint<L0_MVTU_InP*L0_Ibit> > swu_out_reduced("swu_out_reduced");
	ReduceWidth<L0_Cin*L0_Ibit, L0_MVTU_InP*L0_Ibit, L0_K*L0_K*L0_Din*L0_Din> (swu_stream, swu_out_reduced, 1);

	stream<ap_uint<L0_MVTU_OutP*L0_Mbit> > out_raw("out_raw");







    return 0;
}
