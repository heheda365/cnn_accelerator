#include <iostream>
#include <fstream>
#include "conv2d.h"
#include "linear.h"
#include "functional.h"
#include "config.h"
#include "params.h"
#include "loader.h"

void load_data(const char * path, char * ptr, unsigned int size) {
    std::ifstream f(path, std::ios::in | std::ios::binary);
    f.read(ptr, size);
    f.close();
}
void load_params() {
    load_data("params/conv_0_w.bin", (char *) conv_0_w, sizeof(conv_0_w));
    load_data("params/conv_0_b.bin", (char *) conv_0_b, sizeof(conv_0_b));
    load_data("params/conv_1_w.bin", (char *) conv_1_w, sizeof(conv_1_w));
    load_data("params/conv_1_b.bin", (char *) conv_1_b, sizeof(conv_1_b));
    load_data("params/conv_2_w.bin", (char *) conv_2_w, sizeof(conv_2_w));
    load_data("params/conv_2_b.bin", (char *) conv_2_b, sizeof(conv_2_b));
    load_data("params/linear_0_w.bin", (char *) linear_0_w, sizeof(linear_0_w));
    load_data("params/linear_0_b.bin", (char *) linear_0_b, sizeof(linear_0_b));
    load_data("params/linear_1_w.bin", (char *) linear_1_w, sizeof(linear_1_w));
    load_data("params/linear_1_b.bin", (char *) linear_1_b, sizeof(linear_1_b));
}

int main(int argc, char const *argv[])
{
    
    load_params();
    std::cout << "load params finish \n";
    
    float in0[1][28][28];
    load_data("data/test_data.bin", (char *) in0, sizeof(in0));
    std::cout << "load test data finish \n";
    // 神经网络
    float out0[COV_0_OUT_CH][COV_0_OUT_ROW][COV_0_OUT_COL] = {0};
    conv2d<
            COV_0_IN_CH, 
            COV_0_IN_ROW,
            COV_0_IN_COL,
            COV_0_OUT_CH,
            COV_0_OUT_ROW,
            COV_0_OUT_COL,
            COV_0_K,
            COV_0_S,
            COV_0_P,
            COV_0_B>(
                in0,
                out0,
                conv_0_w,
                conv_0_b
            );
    conv_relu<COV_0_OUT_CH, COV_0_OUT_ROW, COV_0_OUT_COL>(out0, out0);

    float out1[COV_1_OUT_CH][COV_1_OUT_ROW][COV_1_OUT_COL] = {0};
    conv2d<
            COV_1_IN_CH, 
            COV_1_IN_ROW,
            COV_1_IN_COL,
            COV_1_OUT_CH,
            COV_1_OUT_ROW,
            COV_1_OUT_COL,
            COV_1_K,
            COV_1_S,
            COV_1_P,
            COV_1_B>(
                out0,
                out1,
                conv_1_w,
                conv_1_b
            );
    conv_relu<COV_1_OUT_CH, COV_1_OUT_ROW, COV_1_OUT_COL>(out1, out1);
    
    float pool_0_out[POOL_0_IN_CH][POOL_0_IN_ROW / POOL_0_IN_PO][POOL_0_IN_COL / POOL_0_IN_PO];
    max_pool2d<
                POOL_0_IN_CH, 
                POOL_0_IN_ROW,
                POOL_0_IN_COL,
                POOL_0_IN_PO> (
                    out1,
                    pool_0_out
                );
    
    float out2[COV_2_OUT_CH][COV_2_OUT_ROW][COV_2_OUT_COL] = {0};
    conv2d<
            COV_2_IN_CH, 
            COV_2_IN_ROW,
            COV_2_IN_COL,
            COV_2_OUT_CH,
            COV_2_OUT_ROW,
            COV_2_OUT_COL,
            COV_2_K,
            COV_2_S,
            COV_2_P,
            COV_2_B>(
                pool_0_out,
                out2,
                conv_2_w,
                conv_2_b
            );
    conv_relu<COV_2_OUT_CH, COV_2_OUT_ROW, COV_2_OUT_COL>(out2, out2);

    float linear_in[LINEAR_0_IN_N] = {0};
    view<COV_2_OUT_CH, COV_2_OUT_ROW, COV_2_OUT_COL>(out2, linear_in);

    float linear_0_out[LINEAR_0_OUT_N] = {0};
    linear<LINEAR_0_IN_N, LINEAR_0_OUT_N>(linear_in, linear_0_out, linear_0_w, linear_0_b);
    linear_relu<LINEAR_0_OUT_N>(linear_0_out, linear_0_out);

    float linear_1_out[LINEAR_1_OUT_N] = {0};
    linear<LINEAR_1_IN_N, LINEAR_1_OUT_N>(linear_0_out, linear_1_out, linear_1_w, linear_1_b);

    float res[LINEAR_1_OUT_N] = {0};
    log_softmax<LINEAR_1_OUT_N>(linear_1_out, res);

    std::cout << std::endl << "res = " << std::endl;
    for(int i=0; i < LINEAR_1_OUT_N; i ++) {
        std::cout << res[i] << "  ";
    }

    int res_num = 0;
    float max = res[0];
    for(int i=0; i < LINEAR_1_OUT_N; i ++) {
        if(max < res[i]) {
            max = res[i];
            res_num = i;
        }
    }
    std::cout << std::endl << "the numble is " << res_num << std::endl;

    return 0;
}
