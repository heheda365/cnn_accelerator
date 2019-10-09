#include <iostream>
#include "conv2d.h"
#include "linear.h"
#include "functional.h"
#include "params.h"
#include "config.h"
#include "loader.h"

int main(int argc, char const *argv[])
{
    std::cout << "hello world";
    loader load = loader();

    load.load_libsvm_data("../load_dataset/MNIST/mnist.t", 1, 784, 10);
	load.x_normalize(0, 'r');
    int cnt = 0;
    float in0[1][28][28];
    for(int i=0; i<28; i ++) {
        for(int j=0; j < 28; j ++) {
            // std::cout << load.x[cnt++];
            in0[0][i][j] = load.x[cnt];
            printf("%4d", (int)(load.x[cnt] * 255));
            cnt ++;
        }
        std::cout << endl;
    }
    for(int i=0; i<10; i ++) {
        std::cout << (int)load.y[i];
    }
     
    // 神经网络
    float out0[COV_0_OUT_CH][COV_0_OUT_ROW][COV_0_OUT_COL];
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

    float out1[COV_1_OUT_CH][COV_1_OUT_ROW][COV_1_OUT_COL];
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
    
    float out2[COV_2_OUT_CH][COV_2_OUT_ROW][COV_2_OUT_COL];
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

    float linear_in[LINEAR_0_IN_N];
    view<COV_2_OUT_CH, COV_2_OUT_ROW, COV_2_OUT_COL>(out2, linear_in);

    float linear_0_out[LINEAR_0_OUT_N];
    linear<LINEAR_0_IN_N, LINEAR_0_OUT_N>(linear_in, linear_0_out, linear_0_w, linear_0_b);

    float linear_1_out[LINEAR_1_OUT_N];
    linear<LINEAR_1_IN_N, LINEAR_1_OUT_N>(linear_0_out, linear_1_out, linear_1_w, linear_1_b);

    float res[LINEAR_1_OUT_N];
    softmax<LINEAR_1_OUT_N>(linear_1_out, res);

    std::cout << "res = " << std::endl;
    for(int i=0; i < LINEAR_1_OUT_N; i ++) {
        std::cout << res[i] << "  ";
    }

    return 0;
}
